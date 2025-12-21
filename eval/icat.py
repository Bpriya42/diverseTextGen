"""
ICAT (Coverage and Factuality) Evaluation

Implements the ICAT-A (Automatic) evaluation methodology.
"""

import os
import json
import torch
import time
import argparse
import logging
import re
import numpy as np

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config.settings import CORPUS_PATH, NLI_MODEL_NAME, NLI_BATCH_SIZE, LLM_BATCH_SIZE, SERVER_LOG_FILE
from eval.retriever import Retriever
from eval.llm_evaluator import LLMEvaluator


class ICAT:
    """
    ICAT Evaluation System.
    
    Evaluates model responses for coverage and factuality.
    """
    
    def __init__(
        self,
        nli_model_name: str = None,
        corpus_path: Optional[str] = None,
        qrels_path: Optional[str] = None,
        queries_path: Optional[str] = None,
        nli_batch_size: int = None,
        llm_batch_size: int = None,
        api_base_llm: str = None,
        api_facts_llm: str = None,
        use_web_search: bool = False,
        hf_token: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        cache_path: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        vllm_logging_level: Optional[str] = None
    ):
        # Use config defaults
        nli_model_name = nli_model_name or NLI_MODEL_NAME
        nli_batch_size = nli_batch_size or NLI_BATCH_SIZE
        llm_batch_size = llm_batch_size or LLM_BATCH_SIZE
        corpus_path = corpus_path or CORPUS_PATH
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Set environment variables if provided
        if hf_token:
            os.environ['HF_TOKEN'] = hf_token
        if brave_api_key:
            os.environ['BRAVE_API_KEY'] = brave_api_key
        if cache_path:
            os.environ['TRANSFORMERS_CACHE'] = cache_path
            os.environ['HF_HOME'] = cache_path
            os.environ['HF_DATASETS_CACHE'] = cache_path
            os.environ['TORCH_HOME'] = cache_path
        
        self.logger.info("Initializing ICAT...")
        self.use_web_search = use_web_search
        self.topk = 10

        if not use_web_search:
            if corpus_path is None:
                raise ValueError("corpus_path must be provided when not using web search")
            self.retriever = Retriever(hf_token=hf_token)
            self.retriever.process_corpus(corpus_path)
        
        # Use server log file path from config (same as agents)
        self.llm_evaluator = LLMEvaluator(
            server_log_path=SERVER_LOG_FILE,
            api_base_llm=api_base_llm,
            api_facts_llm=api_facts_llm,
            hf_token=hf_token,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            vllm_logging_level=vllm_logging_level,
            cache_path=cache_path
        )
        self.qrels_lookup = {}
        
        # Make qrels loading optional
        self.qrels = None
        if qrels_path:
            self.logger.info(f"Loading qrels from {qrels_path}")
            with open(qrels_path, "r") as f:
                self.qrels = [json.loads(line) for line in f]
                for qrel in self.qrels:
                    if qrel["relevance"] == 1:
                        if qrel["doc_id"] not in self.qrels_lookup:
                            self.qrels_lookup[qrel["doc_id"]] = []
                        self.qrels_lookup[qrel["doc_id"]].append(qrel["subtopic_id"])

        # Initialize NLI model
        self.device = torch.device("cpu")
        try:
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                self.device = torch.device("cuda")
                self.logger.info("Attempting to use GPU for NLI model...")
            else:
                self.logger.info("CUDA not available, using CPU for NLI model")
        except Exception as e:
            self.logger.warning(f"GPU initialization failed ({e}), falling back to CPU")
            self.device = torch.device("cpu")
        
        try:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
            self.nli_model.eval()
            
            if self.device.type == 'cuda':
                try:
                    test_inputs = self.nli_tokenizer(["test"], ["test"], return_tensors="pt", padding=True, truncation=True)
                    test_inputs = {k: v.to(self.device) for k, v in test_inputs.items()}
                    with torch.no_grad():
                        _ = self.nli_model(**test_inputs)
                    self.logger.info("✓ GPU verified - using GPU for NLI model")
                except Exception as e:
                    self.logger.warning(f"GPU NLI model test failed ({e}), falling back to CPU")
                    self.device = torch.device("cpu")
                    self.nli_model = self.nli_model.to('cpu')
                    torch.cuda.empty_cache()
                    self.logger.info("Using CPU for NLI model")
            else:
                self.logger.info("Using CPU for NLI model")
        except Exception as e:
            self.logger.warning(f"NLI model initialization failed on GPU ({e}), falling back to CPU")
            self.device = torch.device("cpu")
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
            self.nli_model.eval()
            self.logger.info("Using CPU for NLI model")

        self.nli_batch_size = nli_batch_size
        self.llm_batch_size = llm_batch_size

        # Load queries if provided
        self.queries_data = []
        self.query_id_map = {}
        if queries_path:
            self.logger.info(f"Loading queries from {queries_path}")
            import jsonlines
            with jsonlines.open(queries_path) as reader:
                for obj in reader:
                    query_data = {
                        'query_id': obj['query_id'],
                        'query': obj['query'],
                        'subtopics': obj['subtopics']
                    }
                    self.queries_data.append(query_data)
                    self.query_id_map[obj['query_id']] = query_data
        else:
            self.logger.info("No queries_path provided - queries will be provided directly")

    def _retrieve_documents(self, query: str) -> List[Tuple[str, str, float]]:
        """Unified retrieval method."""
        if self.use_web_search:
            results = self._brave_search(query)
            return [(title, f"{title} {snippet}", 1.0) for title, snippet in results]
        else:
            return self.retriever.retrieve(query, top_k=self.topk)

    def _check_entailment_batch(self, premises: List[str], hypotheses: List[str]) -> List[bool]:
        """Process multiple premise-hypothesis pairs at once."""
        if not premises or not hypotheses:
            return [False] * len(premises)
        
        valid_pairs = [(p, h) for p, h in zip(premises, hypotheses) if p.strip() and h.strip()]
        
        if not valid_pairs:
            return [False] * len(premises)
        
        valid_premises, valid_hypotheses = zip(*valid_pairs)
        
        try:
            inputs = self.nli_tokenizer(
                list(valid_premises),
                list(valid_hypotheses),
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.nli_model(**inputs)
            
            predictions = torch.softmax(output.logits, -1).cpu().numpy()
            valid_results = [bool(pred[0] > 0.5) for pred in predictions]
        except Exception as e:
            if self.device.type == 'cuda':
                self.logger.warning(f"GPU entailment check failed ({e}), retrying on CPU...")
                self.device = torch.device("cpu")
                self.nli_model = self.nli_model.to('cpu')
                torch.cuda.empty_cache()
                
                inputs = self.nli_tokenizer(
                    list(valid_premises),
                    list(valid_hypotheses),
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output = self.nli_model(**inputs)
                
                predictions = torch.softmax(output.logits, -1).cpu().numpy()
                valid_results = [bool(pred[0] > 0.5) for pred in predictions]
            else:
                raise
        
        final_results = []
        valid_idx = 0
        for p, h in zip(premises, hypotheses):
            if p.strip() and h.strip():
                final_results.append(valid_results[valid_idx])
                valid_idx += 1
            else:
                final_results.append(False)
            
        return final_results

    def icat_score_a(
        self,
        model_responses: List[str],
        query_ids: Optional[List[str]] = None,
        queries: Optional[List[str]] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Calculate ICAT-A (Automatic) scores.
        
        Args:
            model_responses: List of model response strings
            query_ids: Optional list of query IDs
            queries: Optional list of query strings
            
        Returns:
            Tuple of (per-query results, aggregate metrics)
        """
        assert len(model_responses) == len(query_ids) if query_ids is not None else True
        assert len(model_responses) == len(queries) if queries is not None else True
        
        if query_ids is None or queries is None:
            query_ids = [q['query_id'] for q in self.queries_data]
            queries = [q['query'] for q in self.queries_data]
        
        self.logger.info(f"Processing {len(queries)} queries")

        if self.query_id_map:
            query_data = [self.query_id_map[qid] for qid in query_ids]
        else:
            query_data = [
                {'query_id': qid, 'query': query}
                for qid, query in zip(query_ids, queries)
            ]
        
        # Generate topics
        self.logger.info("Generating topics for queries...")
        topics_prompts = [
            f'given this query "{query}" generate all the possible subtopics or related queries from most important to least, up to 10, one in each line with this jsonl format {{"topic": ...}}, nothing else in your response'
            for query in queries
        ]
        generated_topics_raw = self.llm_evaluator.generate(topics_prompts)
        
        self.logger.info("Parsing generated topics...")
        all_generated_topics = []
        for query_idx, topics_raw in enumerate(generated_topics_raw):
            generated_topics = []
            for line in topics_raw.split('\n'):
                if line.strip().startswith('{"topic":'):
                    try:
                        topic = json.loads(line.strip())["topic"]
                        generated_topics.append(topic)
                    except json.JSONDecodeError:
                        continue
            all_generated_topics.append(generated_topics)
            self.logger.info(f"Query {query_idx + 1}: Generated {len(generated_topics)} topics")

        # Get atomic facts (batched to avoid overwhelming the server)
        self.logger.info("Generating atomic facts from responses...")
        all_facts = []
        for i in range(0, len(model_responses), self.llm_batch_size):
            batch_responses = model_responses[i:i + self.llm_batch_size]
            batch_facts = self.llm_evaluator.generate_facts(batch_responses)
            all_facts.extend(batch_facts)
            self.logger.info(f"Processed atomic facts for {min(i + self.llm_batch_size, len(model_responses))}/{len(model_responses)} responses")
        
        # Process entailment
        all_entailed_facts = []
        for query_idx, (query, response_facts) in enumerate(zip(query_data, all_facts)):
            self.logger.info(f"Processing entailment for query {query_idx + 1}")
            entailed_facts = []
            
            for i in range(0, len(response_facts), self.nli_batch_size):
                batch_facts = response_facts[i:i + self.nli_batch_size]
                batch_results = []
                
                for fact in batch_facts:
                    top_docs = self._retrieve_documents(fact)
                    premises = [doc[1] for doc in top_docs]
                    hypotheses = [fact] * len(premises)
                    
                    entailment_results = self._check_entailment_batch(premises, hypotheses)
                    
                    if any(entailment_results):
                        for is_entailed, doc in zip(entailment_results, top_docs):
                            if is_entailed:
                                batch_results.append((doc[0], len(entailed_facts), fact))
                                break
                
                entailed_facts.extend(batch_results)
            
            all_entailed_facts.append(entailed_facts)

        # Calculate coverage
        coverage_prompts = []
        for query_idx, (query, response_facts) in enumerate(zip(query_data, all_entailed_facts)):
            entailed_text = "\n".join([f"{i+1}. {fact}" for i, fact in enumerate([f[2] for f in response_facts])])
            
            coverage_prompt = (
                f'given this query "{query}", the following list of subtopics:\n\n' +
                "\n".join([f"{j+1} : {topic}" for j, topic in enumerate(all_generated_topics[query_idx])]) + "\n\n" +
                f'return the subtopics that are covered in the given text below with a list of facts, '
                f'mention each subtopic only once with a list of fact numbers for each subtopic, '
                f'the fact numbers should reference the most relevant facts that support the subtopic, '
                f'they should be explicitly mentioned in the given text, if they are not explicitly mentioned '
                f'don\'t include them in your response, if some subtopic is not covered without any evidence '
                f'don\'t include it in your response, use this jsonl format '
                f'{{"topic_id": ..., "evidence": [fact_number, ...]}}, one json object per line, '
                f'here is the text with enumerated facts:\n\n{entailed_text}'
            )
            coverage_prompts.append(coverage_prompt)

        covered_topics_responses = []
        for i in range(0, len(coverage_prompts), self.llm_batch_size):
            batch_prompts = coverage_prompts[i:i + self.llm_batch_size]
            batch_responses = self.llm_evaluator.generate(batch_prompts)
            covered_topics_responses.extend(batch_responses)

        results = []
        for query_idx, (query, response_facts, covered_topics_raw) in enumerate(zip(query_data, all_entailed_facts, covered_topics_responses)):
            total_topics = len(all_generated_topics[query_idx])
            covered_data = []
            seen_topic_ids = set()
            
            query_atomic_facts = all_facts[query_idx]
            total_facts = len(query_atomic_facts)
            
            for line in covered_topics_raw.split('\n'):
                if line.strip().startswith('{"topic_id":'):
                    try:
                        data = json.loads(line.strip())
                        if data.get("topic_id") is None:
                            continue
                            
                        try:
                            topic_id = int(data["topic_id"]) - 1
                        except (ValueError, TypeError):
                            continue
                        
                        if not data.get("evidence"):
                            continue
                        
                        if (0 <= topic_id < total_topics) and (topic_id not in seen_topic_ids):
                            valid_evidence = []
                            for fact_num in data["evidence"]:
                                try:
                                    fact_idx = int(fact_num) - 1
                                    if 0 <= fact_idx < total_facts:
                                        valid_evidence.append(fact_idx)
                                except (ValueError, TypeError):
                                    continue
                            
                            if valid_evidence:
                                seen_topic_ids.add(topic_id)
                                covered_data.append({
                                    "topic_id": topic_id + 1,
                                    "evidence": valid_evidence
                                })
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue

            coverage = len(covered_data) / total_topics if total_topics > 0 else 0
            factuality = len(response_facts) / total_facts if total_facts > 0 else 0
            f1 = 2 * (factuality * coverage) / (factuality + coverage) if (factuality + coverage) > 0 else 0

            self.logger.info(f"Query {query_idx + 1} metrics - Coverage: {coverage:.2f}, Factuality: {factuality:.2f}, F1: {f1:.2f}")

            results.append({
                "query_id": query['query_id'],
                "query": query['query'],
                "generated_topics": all_generated_topics[query_idx],
                "atomic_facts": query_atomic_facts,
                "entailed_facts": [f[2] for f in response_facts],
                "covered_topics": covered_data,
                "metrics": {
                    "coverage": coverage,
                    "factuality": factuality,
                    "f1": f1
                }
            })
        
        avg_coverage = sum(r["metrics"]["coverage"] for r in results) / len(results)
        avg_factuality = sum(r["metrics"]["factuality"] for r in results) / len(results)
        avg_f1 = sum(r["metrics"]["f1"] for r in results) / len(results)
        
        aggregate_metrics = {
            "avg_coverage": avg_coverage,
            "avg_factuality": avg_factuality,
            "avg_f1": avg_f1
        }
        
        return results, aggregate_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ICAT evaluation')
    parser.add_argument('--corpus-path', default=None, help='Path to corpus file')
    parser.add_argument('--queries-path', default=None, help='Path to queries file')
    parser.add_argument('--qrels-path', default=None, help='Optional path to qrels file')
    parser.add_argument('--results-file', default='icat_results.txt', help='Path to results file')
    args = parser.parse_args()

    print("Initializing ICAT evaluator...")
    scorer = ICAT(
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        qrels_path=args.qrels_path
    )
    
    print("ICAT evaluator initialized successfully!")

