"""
Dense Retriever

FAISS-based dense retrieval using Sentence Transformers.
"""

import json
import pickle
import faiss
import nltk
import numpy as np
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from config.settings import CACHE_DIR, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE


class Retriever:
    """
    Dense retriever using Sentence Transformers and FAISS.
    
    Args:
        model_name: Sentence transformer model name
        cache_dir: Directory for caching embeddings and indices
        batch_size: Batch size for embedding computation
        hf_token: Optional HuggingFace token
    """
    
    def __init__(
        self,
        model_name: str = None,
        cache_dir: str = None,
        batch_size: int = None,
        hf_token: Optional[str] = None
    ):
        # Use config defaults if not provided
        model_name = model_name or EMBEDDING_MODEL
        cache_dir = cache_dir or CACHE_DIR
        batch_size = batch_size or EMBEDDING_BATCH_SIZE
        
        if hf_token:
            import os
            os.environ['HF_TOKEN'] = hf_token
        
        # Try GPU first, fallback to CPU
        self.device = 'cpu'
        try:
            import torch
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                self.device = 'cuda'
                print("Attempting to use GPU for SentenceTransformer...")
            else:
                print("CUDA not available, using CPU for SentenceTransformer")
        except Exception as e:
            print(f"GPU initialization failed ({e}), falling back to CPU for SentenceTransformer")
            self.device = 'cpu'
        
        # Initialize model
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            if self.device == 'cuda':
                try:
                    test_encoding = self.model.encode(["test"], show_progress_bar=False, convert_to_numpy=True)
                    print("✓ GPU verified - using GPU for SentenceTransformer")
                except Exception as e:
                    print(f"GPU encoding test failed ({e}), falling back to CPU")
                    self.device = 'cpu'
                    self.model = SentenceTransformer(model_name, device='cpu')
                    print("Using CPU for SentenceTransformer")
            else:
                print("Using CPU for SentenceTransformer")
        except Exception as e:
            print(f"Model initialization failed on GPU ({e}), falling back to CPU")
            self.device = 'cpu'
            self.model = SentenceTransformer(model_name, device='cpu')
            print("Using CPU for SentenceTransformer")
            
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.snippets = []
        self.doc_ids = []
        
        try:
            nltk.download('punkt', quiet=True)
        except:
            pass
        
    def create_snippets(self, text: str, max_length: int = 512, stride: int = 96) -> List[str]:
        """Create snippets of up to max_length words."""
        words = [w for w in text.split() if w]
        return [" ".join(words[0:min(max_length, len(words))])]
    
    def process_corpus(self, corpus_path: str):
        """
        Process corpus and create FAISS index.
        
        Args:
            corpus_path: Path to JSONL corpus file
        """
        print(f"Processing corpus from: {corpus_path}")
        snippets_cache_file = self.cache_dir / "snippets_embeddings_large.pkl"
        index_cache_file = self.cache_dir / "faiss_index_large.bin"

        # Load or create embeddings
        if snippets_cache_file.exists():
            print("Loading cached embeddings...")
            with open(snippets_cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.snippets = cached_data['snippets']
                self.doc_ids = cached_data['doc_ids']
                embeddings = cached_data['embeddings'].astype(np.float16)
            print(f"Loaded {len(self.snippets)} snippets from cache")
        else:
            print("Cache not found. Processing corpus...")
            all_snippets = []
            all_doc_ids = []
            
            with open(corpus_path, 'r') as f:
                for i, line in enumerate(f, 1):
                    if i % 1000 == 0:
                        print(f"Processed documents: {i}...")
                    doc = json.loads(line)
                    doc_snippets = self.create_snippets(doc['text'])
                    all_snippets.extend(doc_snippets)
                    all_doc_ids.extend([doc['doc_id']] * len(doc_snippets))
            
            print(f"Calculating embeddings for {len(all_snippets)} snippets...")
            try:
                embeddings = self.model.encode(
                    all_snippets, show_progress_bar=True, batch_size=self.batch_size
                )
            except Exception as e:
                if self.device == 'cuda':
                    print(f"GPU encoding failed ({e}), retrying on CPU...")
                    self.device = 'cpu'
                    self.model = self.model.to('cpu')
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    embeddings = self.model.encode(
                        all_snippets, show_progress_bar=True, batch_size=self.batch_size
                    )
                    print("Successfully encoded on CPU")
                else:
                    raise
                    
            embeddings = embeddings.astype(np.float16)
            
            print("Caching results...")
            self.snippets = all_snippets
            self.doc_ids = all_doc_ids
            cache_data = {
                'snippets': self.snippets,
                'doc_ids': self.doc_ids,
                'embeddings': embeddings
            }
            with open(snippets_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        
        # Load or create FAISS index
        if index_cache_file.exists():
            print("Loading cached index...")
            self.index = faiss.read_index(str(index_cache_file))
            print(f"Loaded index from cache")
        else:
            embeddings = embeddings.astype(np.float32)
            
            print("Building FAISS index...")
            dimension = embeddings.shape[1]
            num_snippets = len(self.snippets)

            use_gpu = False
            if self.device == 'cuda':
                try:
                    use_gpu = faiss.get_num_gpus() > 0
                except Exception as e:
                    print(f"FAISS GPU check failed ({e}), using CPU for index")
                    use_gpu = False
            
            faiss.normalize_L2(embeddings)
            
            if num_snippets > 50000:
                num_centroids = 8 * int(math.sqrt(math.pow(2, int(math.log(num_snippets, 2)))))
                print(f"Using {num_centroids} centroids for {num_snippets} snippets")
                self.index = faiss.index_factory(dimension, f"IVF{num_centroids}_HNSW32,Flat")
                
                if use_gpu:
                    try:
                        print(f"Using {faiss.get_num_gpus()} GPUs for index building...")
                        index_ivf = faiss.extract_index_ivf(self.index)
                        clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(dimension))
                        index_ivf.clustering_index = clustering_index
                    except Exception as e:
                        print(f"FAISS GPU index building failed ({e}), using CPU instead")
                        use_gpu = False
                
                print("Training index...")
                self.index.train(embeddings.astype(np.float32))
            else:
                print("Using simple FlatL2 index for small dataset...")
                self.index = faiss.IndexFlatL2(dimension)
                if use_gpu:
                    try:
                        self.index = faiss.index_cpu_to_all_gpus(self.index)
                    except Exception as e:
                        print(f"FAISS GPU index creation failed ({e}), using CPU instead")
            
            print("Adding vectors to index...")
            self.index.add(embeddings.astype(np.float32))
            
            try:
                self.index.nprobe = 256
            except:
                pass
            
            print("Caching FAISS index...")
            faiss.write_index(self.index, str(index_cache_file))
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Retrieve top-k relevant snippets.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (doc_id, snippet_text, score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call process_corpus first.")
        
        try:
            query_embedding = self.model.encode([query])[0]
        except Exception as e:
            if self.device == 'cuda':
                print(f"GPU query encoding failed ({e}), retrying on CPU...")
                self.device = 'cpu'
                self.model = self.model.to('cpu')
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                query_embedding = self.model.encode([query])[0]
            else:
                raise
                
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Convert L2 distances to similarities
        similarities = [(2 - d) / 2 for d in distances[0]]
        
        results = []
        for similarity, idx in zip(similarities, indices[0]):
            results.append((
                self.doc_ids[idx],
                self.snippets[idx],
                float(similarity)
            ))
        
        return results


if __name__ == "__main__":
    from config.settings import CORPUS_PATH
    retriever = Retriever()
    retriever.process_corpus(CORPUS_PATH)
    
    # Test query
    results = retriever.retrieve("What causes headaches?", top_k=3)
    for doc_id, text, score in results:
        print(f"[{score:.3f}] {doc_id}: {text[:100]}...")

