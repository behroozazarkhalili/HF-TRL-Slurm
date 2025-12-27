#!/usr/bin/env python3
"""
Model Evaluation Script with lm-eval-harness
=============================================
Comprehensive evaluation across multiple benchmark categories:
- Reasoning: GSM8K, MATH, ARC-Challenge, ARC-Easy
- General: MMLU, HellaSwag, TruthfulQA, Winogrande
- Coding: HumanEval, MBPP

Usage:
    python evaluate_model.py \
        --model username/my-model \
        --tasks comprehensive \
        --output_dir ./eval_results \
        --push_to_hub
"""

import os
import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict

# Benchmark definitions
BENCHMARK_SUITES = {
    "reasoning": [
        "gsm8k",
        "minerva_math",
        "arc_challenge",
        "arc_easy",
    ],
    "general": [
        "mmlu",
        "hellaswag",
        "truthfulqa_mc2",
        "winogrande",
    ],
    "coding": [
        "humaneval",
        "mbpp",
    ],
    "comprehensive": [
        # Reasoning
        "gsm8k",
        "arc_challenge",
        "arc_easy",
        # General
        "mmlu",
        "hellaswag",
        "truthfulqa_mc2",
        "winogrande",
        # Coding
        "humaneval",
        "mbpp",
    ],
    "quick": [
        "arc_easy",
        "hellaswag",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model with lm-eval-harness")

    # Model arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Model to evaluate (HF model ID or local path)")
    parser.add_argument("--model_type", type=str, default="hf",
                        choices=["hf", "vllm", "hf-multimodal"],
                        help="Model backend type")
    parser.add_argument("--revision", type=str, default=None,
                        help="Model revision/branch")

    # Task arguments
    parser.add_argument("--tasks", type=str, nargs="+", required=True,
                        help="Benchmark suites or individual tasks")
    parser.add_argument("--num_fewshot", type=int, default=None,
                        help="Number of few-shot examples (task-dependent if not set)")

    # Generation arguments
    parser.add_argument("--batch_size", type=str, default="auto",
                        help="Batch size (auto, or integer)")
    parser.add_argument("--max_batch_size", type=int, default=64,
                        help="Maximum batch size for auto-batching")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")

    # Chat template arguments
    parser.add_argument("--apply_chat_template", action="store_true",
                        help="Apply chat template to prompts")
    parser.add_argument("--system_instruction", type=str, default=None,
                        help="System instruction for chat template")

    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--output_format", type=str, default="json",
                        choices=["json", "csv", "markdown"],
                        help="Output format")

    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push results to model card on HF Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Hub model ID (defaults to --model)")

    # Logging
    parser.add_argument("--verbosity", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity")

    return parser.parse_args()


def expand_task_list(task_args: List[str]) -> List[str]:
    """Expand task suite names into individual tasks."""
    tasks = []
    for task_arg in task_args:
        if task_arg in BENCHMARK_SUITES:
            tasks.extend(BENCHMARK_SUITES[task_arg])
        else:
            tasks.append(task_arg)
    # Remove duplicates while preserving order
    seen = set()
    unique_tasks = []
    for task in tasks:
        if task not in seen:
            seen.add(task)
            unique_tasks.append(task)
    return unique_tasks


def run_evaluation(args) -> Dict:
    """Run lm-eval-harness evaluation."""
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    print("=" * 60)
    print("Model Evaluation with lm-eval-harness")
    print("=" * 60)

    # Expand tasks
    tasks = expand_task_list(args.tasks)
    print(f"\nEvaluating tasks: {', '.join(tasks)}")

    # Model arguments
    model_args = f"pretrained={args.model}"
    if args.revision:
        model_args += f",revision={args.revision}"

    # Build lm_eval arguments
    eval_args = {
        "model": args.model_type,
        "model_args": model_args,
        "tasks": tasks,
        "batch_size": args.batch_size,
        "device": args.device,
        "log_samples": True,
    }

    if args.num_fewshot is not None:
        eval_args["num_fewshot"] = args.num_fewshot

    if args.apply_chat_template:
        eval_args["apply_chat_template"] = True
        if args.system_instruction:
            eval_args["system_instruction"] = args.system_instruction

    # Run evaluation
    print("\nStarting evaluation...")
    print(f"Model: {args.model}")
    print(f"Tasks: {len(tasks)} benchmarks")
    print("")

    results = evaluator.simple_evaluate(**eval_args)

    return results


def format_results(results: Dict, output_format: str) -> str:
    """Format results for output."""
    if output_format == "json":
        return json.dumps(results, indent=2, default=str)

    elif output_format == "markdown":
        lines = ["# Evaluation Results\n"]
        lines.append(f"**Model:** {results.get('config', {}).get('model', 'Unknown')}\n")
        lines.append(f"**Date:** {datetime.now().isoformat()}\n\n")

        lines.append("## Benchmark Results\n")
        lines.append("| Task | Metric | Value |")
        lines.append("|------|--------|-------|")

        for task_name, task_results in results.get("results", {}).items():
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    if "acc" in metric_name:
                        lines.append(f"| {task_name} | {metric_name} | {metric_value:.4f} |")

        return "\n".join(lines)

    elif output_format == "csv":
        lines = ["task,metric,value"]
        for task_name, task_results in results.get("results", {}).items():
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    lines.append(f"{task_name},{metric_name},{metric_value}")
        return "\n".join(lines)


def save_results(results: Dict, output_dir: str, output_format: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Always save full JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Full results saved to: {json_path}")

    # Save formatted output
    if output_format != "json":
        formatted = format_results(results, output_format)
        ext = {"markdown": "md", "csv": "csv"}[output_format]
        formatted_path = os.path.join(output_dir, f"results.{ext}")
        with open(formatted_path, "w") as f:
            f.write(formatted)
        print(f"Formatted results saved to: {formatted_path}")


def push_results_to_hub(results: Dict, model_id: str):
    """Push evaluation results to model card on HF Hub."""
    try:
        from huggingface_hub import HfApi, ModelCard

        api = HfApi()

        # Create results table
        table_rows = []
        for task_name, task_results in results.get("results", {}).items():
            acc = task_results.get("acc,none", task_results.get("acc_norm,none", "N/A"))
            if isinstance(acc, float):
                table_rows.append(f"| {task_name} | {acc:.4f} |")

        eval_section = f"""
## Evaluation Results

Evaluated on {datetime.now().strftime("%Y-%m-%d")}

| Task | Accuracy |
|------|----------|
{chr(10).join(table_rows)}

"""

        # Try to update model card
        try:
            card = ModelCard.load(model_id)
            # Append evaluation results
            card.text += eval_section
            card.push_to_hub(model_id)
            print(f"Evaluation results pushed to: https://huggingface.co/{model_id}")
        except Exception as e:
            print(f"Warning: Could not update model card: {e}")
            # Save results as separate file
            results_json = json.dumps(results, indent=2, default=str)
            api.upload_file(
                path_or_fileobj=results_json.encode(),
                path_in_repo="eval_results.json",
                repo_id=model_id,
                repo_type="model",
            )
            print(f"Evaluation results uploaded as eval_results.json")

    except Exception as e:
        print(f"Error pushing to Hub: {e}")


def print_summary(results: Dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    for task_name, task_results in results.get("results", {}).items():
        # Find main accuracy metric
        acc = None
        for key in ["acc,none", "acc_norm,none", "exact_match,none"]:
            if key in task_results:
                acc = task_results[key]
                break

        if acc is not None:
            print(f"  {task_name}: {acc:.4f}")
        else:
            print(f"  {task_name}: (see full results)")

    print("=" * 60)


def main():
    args = parse_args()

    # Check lm-eval installation
    try:
        import lm_eval
    except ImportError:
        print("ERROR: lm-eval not installed.")
        print("Install with: pip install lm-eval[hf]")
        return

    # Run evaluation
    results = run_evaluation(args)

    # Save results
    save_results(results, args.output_dir, args.output_format)

    # Print summary
    print_summary(results)

    # Push to Hub
    if args.push_to_hub:
        hub_model_id = args.hub_model_id or args.model
        push_results_to_hub(results, hub_model_id)

    print("\nEvaluation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
