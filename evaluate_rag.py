#!/usr/bin/env python3
"""
RAG Evaluation Utilities

Tools to measure and compare the quality of different RAG configurations.
"""

import json
import time
from typing import List, Dict
from rag_experiments import RAGConfig, query_with_config


# ============================================================================
# TEST CASES
# ============================================================================

# Define test questions with known correct answers
TEST_CASES = [
    {
        'question': 'When was Abraham Lincoln born?',
        'expected_keywords': ['February 12', '1809', 'February', '12'],
        'category': 'simple_fact'
    },
    {
        'question': 'When was George Washington born?',
        'expected_keywords': ['February 22', '1732', 'February', '22'],
        'category': 'simple_fact'
    },
    {
        'question': 'Who was the first president of the United States?',
        'expected_keywords': ['George Washington', 'Washington'],
        'category': 'simple_fact'
    },
    {
        'question': 'Which president signed the Emancipation Proclamation?',
        'expected_keywords': ['Abraham Lincoln', 'Lincoln'],
        'category': 'simple_fact'
    },
    {
        'question': 'What were Theodore Roosevelt\'s major accomplishments?',
        'expected_keywords': ['Panama Canal', 'conservation', 'trust', 'Nobel'],
        'category': 'complex'
    },
    {
        'question': 'What were the main policies of FDR\'s New Deal?',
        'expected_keywords': ['Social Security', 'relief', 'recovery', 'reform'],
        'category': 'complex'
    },
    {
        'question': 'Which presidents served during World War II?',
        'expected_keywords': ['Franklin', 'Roosevelt', 'Truman', 'FDR'],
        'category': 'multiple_entity'
    },
    {
        'question': 'Compare the foreign policies of Reagan and Kennedy',
        'expected_keywords': ['Soviet', 'Cold War', 'Cuba', 'missile'],
        'category': 'comparison'
    },
]


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def keyword_presence_score(answer: str, keywords: List[str]) -> float:
    """
    Simple metric: what percentage of expected keywords appear in answer?
    
    Args:
        answer: Generated answer
        keywords: Expected keywords
        
    Returns:
        Score from 0.0 to 1.0
    """
    answer_lower = answer.lower()
    present = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return present / len(keywords) if keywords else 0.0


def answer_length_score(answer: str, optimal_min: int = 50, optimal_max: int = 300) -> float:
    """
    Score based on answer length. Too short = incomplete, too long = verbose.
    
    Args:
        answer: Generated answer
        optimal_min: Minimum good length
        optimal_max: Maximum good length
        
    Returns:
        Score from 0.0 to 1.0
    """
    length = len(answer)
    
    if length < optimal_min:
        return length / optimal_min
    elif length > optimal_max:
        penalty = (length - optimal_max) / optimal_max
        return max(0.0, 1.0 - penalty)
    else:
        return 1.0


def retrieval_quality_score(chunks: List[Dict], question: str) -> Dict:
    """
    Metrics about the retrieval quality.
    
    Args:
        chunks: Retrieved chunks
        question: The query
        
    Returns:
        Dict with metrics
    """
    if not chunks:
        return {'avg_distance': float('inf'), 'source_diversity': 0}
    
    # Average distance (lower is better)
    distances = [c.get('distance', 0) for c in chunks]
    avg_distance = sum(distances) / len(distances)
    
    # Source diversity (how many different documents?)
    sources = set(c['metadata'].get('filename', '') for c in chunks)
    source_diversity = len(sources)
    
    return {
        'avg_distance': avg_distance,
        'source_diversity': source_diversity,
        'num_chunks': len(chunks)
    }


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_single_config(config: RAGConfig,
                          test_cases: List[Dict] = None,
                          n_results: int = 5,
                          llm_model: str = "llama3.2") -> Dict:
    """
    Evaluate a single configuration on all test cases.
    
    Args:
        config: RAGConfig to test
        test_cases: List of test case dicts (defaults to TEST_CASES)
        n_results: Number of chunks to retrieve
        llm_model: LLM model to use
        
    Returns:
        Dict with evaluation results
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    print(f"\n{'='*70}")
    print(f"EVALUATING: {config}")
    print(f"{'='*70}\n")
    
    results = []
    total_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"[{i}/{len(test_cases)}] {test_case['question']}")
        
        # Query
        start_time = time.time()
        result = query_with_config(
            test_case['question'],
            config,
            n_results=n_results,
            llm_model=llm_model,
            verbose=False
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if 'error' in result:
            print(f"  ✗ Error: {result['error']}")
            continue
        
        # Evaluate
        answer = result.get('answer', '')
        
        # Keyword score
        kw_score = keyword_presence_score(answer, test_case['expected_keywords'])
        
        # Length score
        length_score = answer_length_score(answer)
        
        # Combined score
        overall_score = (kw_score * 0.7) + (length_score * 0.3)
        
        print(f"  ✓ Keyword: {kw_score:.2f}, Length: {length_score:.2f}, "
              f"Overall: {overall_score:.2f} ({elapsed:.2f}s)")
        
        results.append({
            'question': test_case['question'],
            'category': test_case['category'],
            'answer': answer,
            'keyword_score': kw_score,
            'length_score': length_score,
            'overall_score': overall_score,
            'latency': elapsed,
            'collection_name': result['collection_name']
        })
    
    # Summary statistics
    avg_keyword = sum(r['keyword_score'] for r in results) / len(results)
    avg_length = sum(r['length_score'] for r in results) / len(results)
    avg_overall = sum(r['overall_score'] for r in results) / len(results)
    avg_latency = total_time / len(results)
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Average Keyword Score: {avg_keyword:.3f}")
    print(f"Average Length Score:  {avg_length:.3f}")
    print(f"Average Overall Score: {avg_overall:.3f}")
    print(f"Average Latency:       {avg_latency:.2f}s")
    print(f"{'='*70}\n")
    
    return {
        'config': config.to_dict(),
        'results': results,
        'summary': {
            'avg_keyword_score': avg_keyword,
            'avg_length_score': avg_length,
            'avg_overall_score': avg_overall,
            'avg_latency': avg_latency,
            'total_questions': len(results)
        }
    }


def compare_multiple_configs(configs: List[RAGConfig],
                            test_cases: List[Dict] = None,
                            n_results: int = 5,
                            llm_model: str = "llama3.2") -> List[Dict]:
    """
    Evaluate and compare multiple configurations.
    
    Args:
        configs: List of RAGConfig objects to compare
        test_cases: Test cases to use
        n_results: Number of chunks to retrieve
        llm_model: LLM model
        
    Returns:
        List of evaluation results
    """
    all_results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n\n{'#'*70}")
        print(f"CONFIG {i}/{len(configs)}")
        print(f"{'#'*70}")
        
        result = evaluate_single_config(config, test_cases, n_results, llm_model)
        all_results.append(result)
    
    # Print comparison table
    print(f"\n\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Config':<30} {'Overall':<10} {'Keyword':<10} {'Length':<10} {'Latency':<10}")
    print("-" * 70)
    
    for result in all_results:
        config_name = result['config']['embedding_model'].split('/')[-1][:25]
        summary = result['summary']
        
        print(f"{config_name:<30} "
              f"{summary['avg_overall_score']:<10.3f} "
              f"{summary['avg_keyword_score']:<10.3f} "
              f"{summary['avg_length_score']:<10.3f} "
              f"{summary['avg_latency']:<10.2f}")
    
    print("=" * 70)
    
    # Find best config
    best = max(all_results, key=lambda x: x['summary']['avg_overall_score'])
    print(f"\nBest Config: {best['config']['embedding_model']}")
    print(f"Score: {best['summary']['avg_overall_score']:.3f}")
    
    return all_results


def save_evaluation_results(results: Dict, filename: str = "evaluation_results.json"):
    """Save evaluation results to JSON."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved evaluation to {filename}")


# ============================================================================
# INTERACTIVE EVALUATION
# ============================================================================

def interactive_evaluation():
    """Interactive mode to evaluate configs."""
    from rag_experiments import list_collections
    
    print("\n" + "=" * 70)
    print("INTERACTIVE EVALUATION")
    print("=" * 70)
    
    # Show available collections
    collections = list_collections()
    if not collections:
        print("\nNo collections found. Run indexing first.")
        return
    
    print("\nAvailable collections:")
    for i, col in enumerate(collections, 1):
        print(f"  {i}. {col['name']} ({col['count']} chunks)")
    
    print("\nSelect evaluation mode:")
    print("  1. Quick test (3 questions)")
    print("  2. Full evaluation (all test cases)")
    print("  3. Custom questions")
    
    mode = input("\nChoice (1-3): ").strip()
    
    if mode == '1':
        test_cases = TEST_CASES[:3]
    elif mode == '2':
        test_cases = TEST_CASES
    elif mode == '3':
        test_cases = []
        print("\nEnter questions (empty line to finish):")
        while True:
            q = input("Question: ").strip()
            if not q:
                break
            keywords = input("Expected keywords (comma-separated): ").strip().split(',')
            test_cases.append({
                'question': q,
                'expected_keywords': [kw.strip() for kw in keywords],
                'category': 'custom'
            })
    else:
        print("Invalid choice")
        return
    
    # TODO: Allow selecting configs from collections
    # For now, just show how to use it programmatically
    print("\nTo evaluate specific configs, modify run_experiments.py")
    print("or use evaluate_single_config() directly in code.")


if __name__ == "__main__":
    interactive_evaluation()
