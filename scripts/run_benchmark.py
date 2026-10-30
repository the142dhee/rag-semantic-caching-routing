"""
scripts/run_benchmark.py
========================
PURPOSE: CLI script to execute the Stage 5 RAG Benchmark suite.
Compares BASELINE_RAG against OPTIMIZED_RAG across our 15 evaluation queries.

Usage:
  python scripts/run_benchmark.py [--mock]
"""

import sys
import os
import argparse

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.evaluation.benchmark_runner import BenchmarkRunner
from app.core.config import get_settings

def main():
    parser = argparse.ArgumentParser(description="Run Stage 5 RAG Benchmark")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock execution (useful for offline testing or without OpenAI credits)",
    )
    args = parser.parse_args()

    settings = get_settings()
    use_mock = args.mock or (not settings.openai_api_key or settings.openai_api_key.startswith("your-") or settings.openai_api_key == "sk-test")

    print("\n" + "=" * 60)
    print("STAGE 5: RAG BENCHMARK & EVALUATION RUNNER")
    print(f"Execution Mode: {'MOCK (Deterministic Offline)' if use_mock else 'LIVE (OpenAI API)'}")
    print("=" * 60 + "\n")

    runner = BenchmarkRunner()
    report = runner.run(mock_llm=use_mock)

    comp = report["comparison"]
    base = comp["baseline"]
    opt = comp["optimized"]
    imp = comp["improvements"]

    print("\n" + "=" * 60)
    print("END-TO-END COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'BASELINE':<15} {'OPTIMIZED':<15}")
    print("-" * 60)
    print(f"{'Cache Hit Rate':<30} {str(base['cache_hit_rate_pct']) + '%' :<15} {str(opt['cache_hit_rate_pct']) + '%' :<15}")
    print(f"{'Average Latency (ms)':<30} {str(base['avg_latency_ms']) :<15} {str(opt['avg_latency_ms']) :<15}")
    print(f"{'p50 Latency (ms)':<30} {str(base['p50_latency_ms']) :<15} {str(opt['p50_latency_ms']) :<15}")
    print(f"{'p95 Latency (ms)':<30} {str(base['p95_latency_ms']) :<15} {str(opt['p95_latency_ms']) :<15}")
    print(f"{'LLM Calls Executed':<30} {str(base['llm_calls']) :<15} {str(opt['llm_calls']) :<15}")
    print(f"{'Total Tokens Consumed':<30} {str(base['total_tokens']) :<15} {str(opt['total_tokens']) :<15}")
    print(f"{'Total Estimated Cost ($)':<30} {'$' + str(base['total_cost_usd']) :<15} {'$' + str(opt['total_cost_usd']) :<15}")
    print(f"{'Cost Per Query ($)':<30} {'$' + str(base['cost_per_query_usd']) :<15} {'$' + str(opt['cost_per_query_usd']) :<15}")
    print(f"{'Answer Quality (Token F1)':<30} {str(base['avg_f1_score']) :<15} {str(opt['avg_f1_score']) :<15}")
    print(f"{'Faithfulness Score':<30} {str(base['avg_faithfulness']) :<15} {str(opt['avg_faithfulness']) :<15}")
    print(f"{'Retrieval Precision@K':<30} {str(base['avg_precision_at_k']) :<15} {str(opt['avg_precision_at_k']) :<15}")
    print(f"{'Retrieval Recall@K':<30} {str(base['avg_recall_at_k']) :<15} {str(opt['avg_recall_at_k']) :<15}")
    print("-" * 60)
    print(f"Latency Reduction: {imp['latency_reduction_pct']}%")
    print(f"Cost Reduction:    {imp['cost_reduction_pct']}%")
    print(f"Token Reduction:   {imp['token_reduction_pct']}%")
    print(f"LLM Calls Avoided: {imp['llm_calls_avoided']}")
    print("=" * 60 + "\n")
    print(f"Full results saved to: {runner.results_path}\n")

if __name__ == "__main__":
    main()
