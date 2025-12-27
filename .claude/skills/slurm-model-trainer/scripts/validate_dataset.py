#!/usr/bin/env python3
"""
Dataset Validation Script
=========================
Validate dataset format before training to prevent failures.

Checks compatibility with:
- SFT: messages, text, or prompt+completion format
- DPO: prompt, chosen, rejected columns
- GRPO: prompt column

Usage:
    python validate_dataset.py --dataset trl-lib/Capybara --method sft
    python validate_dataset.py --dataset your/dataset --method dpo
"""

import argparse
import json
from typing import Dict, List, Optional, Any


def parse_args():
    parser = argparse.ArgumentParser(description="Validate dataset for training")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset name or path")
    parser.add_argument("--config", type=str, default=None,
                        help="Dataset configuration")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to validate")
    parser.add_argument("--method", type=str, default="all",
                        choices=["sft", "dpo", "grpo", "kto", "all"],
                        help="Training method to validate for")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of samples to show")
    parser.add_argument("--output", type=str, choices=["human", "json"],
                        default="human", help="Output format")

    return parser.parse_args()


# Column requirements for each method
METHOD_REQUIREMENTS = {
    "sft": {
        "required_any": [
            ["messages"],  # Chat format
            ["text"],      # Text format
            ["prompt", "completion"],  # Prompt-completion
            ["instruction", "output"],  # Instruction format
        ],
        "description": "SFT needs messages/text/prompt+completion format",
    },
    "dpo": {
        "required_all": ["prompt", "chosen", "rejected"],
        "alternative_mappings": {
            "prompt": ["instruction", "question", "input", "query"],
            "chosen": ["chosen_response", "preferred", "chosen_text", "positive"],
            "rejected": ["rejected_response", "dispreferred", "rejected_text", "negative"],
        },
        "description": "DPO requires prompt, chosen, and rejected columns",
    },
    "grpo": {
        "required_all": ["prompt"],
        "alternative_mappings": {
            "prompt": ["instruction", "question", "input", "query"],
        },
        "description": "GRPO requires prompt column",
    },
    "kto": {
        "required_all": ["prompt", "completion", "label"],
        "alternative_mappings": {
            "prompt": ["instruction", "question", "input"],
            "completion": ["response", "output", "answer"],
            "label": ["preference", "rating"],
        },
        "description": "KTO requires prompt, completion, and label columns",
    },
}


def check_method_compatibility(
    columns: List[str],
    method: str,
    sample: Dict[str, Any]
) -> Dict:
    """Check if dataset is compatible with a training method."""
    requirements = METHOD_REQUIREMENTS[method]
    result = {
        "method": method,
        "compatible": False,
        "status": "INCOMPATIBLE",
        "message": "",
        "mapping": None,
    }

    # Check required_any (e.g., SFT)
    if "required_any" in requirements:
        for option in requirements["required_any"]:
            if all(col in columns for col in option):
                result["compatible"] = True
                result["status"] = "READY"
                result["message"] = f"Found required columns: {option}"
                return result

        result["message"] = requirements["description"]
        return result

    # Check required_all with alternative mappings
    if "required_all" in requirements:
        required = requirements["required_all"]
        alternatives = requirements.get("alternative_mappings", {})

        # Check if all required columns exist directly
        if all(col in columns for col in required):
            result["compatible"] = True
            result["status"] = "READY"
            result["message"] = f"Found required columns: {required}"
            return result

        # Check for alternative column names
        mapping = {}
        all_found = True

        for req_col in required:
            if req_col in columns:
                mapping[req_col] = req_col
            elif req_col in alternatives:
                found = False
                for alt in alternatives[req_col]:
                    if alt in columns:
                        mapping[req_col] = alt
                        found = True
                        break
                if not found:
                    all_found = False
                    break
            else:
                all_found = False
                break

        if all_found and mapping:
            result["compatible"] = True
            result["status"] = "NEEDS MAPPING"
            result["message"] = f"Columns can be mapped: {mapping}"
            result["mapping"] = mapping

            # Generate mapping code
            code_lines = ["def format_for_{}(example):".format(method)]
            code_lines.append("    return {")
            for target, source in mapping.items():
                if target != source:
                    code_lines.append(f"        '{target}': example['{source}'],")
                else:
                    code_lines.append(f"        '{target}': example['{target}'],")
            code_lines.append("    }")
            code_lines.append("")
            code_lines.append("dataset = dataset.map(format_for_{}, remove_columns=dataset.column_names)".format(method))
            result["mapping_code"] = "\n".join(code_lines)

            return result

        result["message"] = f"Missing columns: {[c for c in required if c not in columns and c not in [mapping.get(c) for c in required]]}"
        return result

    return result


def validate_dataset(args) -> Dict:
    """Validate dataset and return results."""
    from datasets import load_dataset

    print(f"Loading dataset: {args.dataset}")
    print(f"Config: {args.config or 'default'}")
    print(f"Split: {args.split}")
    print("")

    # Load dataset
    try:
        dataset = load_dataset(
            args.dataset,
            args.config,
            split=args.split,
            streaming=True,  # Use streaming for large datasets
        )
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to load dataset: {e}",
        }

    # Get columns and sample
    sample = next(iter(dataset))
    columns = list(sample.keys())

    print(f"Columns: {columns}")
    print("")

    # Check each method
    methods_to_check = [args.method] if args.method != "all" else ["sft", "dpo", "grpo", "kto"]
    results = {}

    for method in methods_to_check:
        result = check_method_compatibility(columns, method, sample)
        results[method] = result

    # Prepare output
    output = {
        "dataset": args.dataset,
        "columns": columns,
        "sample_keys": list(sample.keys()),
        "results": results,
    }

    return output


def print_human_output(results: Dict, args):
    """Print human-readable output."""
    print("=" * 60)
    print("Dataset Validation Results")
    print("=" * 60)
    print(f"Dataset: {results['dataset']}")
    print(f"Columns: {results['columns']}")
    print("")

    for method, result in results["results"].items():
        status_icon = {
            "READY": "✓",
            "NEEDS MAPPING": "⚠",
            "INCOMPATIBLE": "✗",
        }.get(result["status"], "?")

        print(f"{method.upper()}: {status_icon} {result['status']}")
        print(f"  {result['message']}")

        if result.get("mapping_code"):
            print("")
            print("  MAPPING CODE:")
            print("  " + "-" * 40)
            for line in result["mapping_code"].split("\n"):
                print(f"  {line}")
            print("  " + "-" * 40)

        print("")


def print_json_output(results: Dict):
    """Print JSON output."""
    print(json.dumps(results, indent=2))


def main():
    args = parse_args()

    try:
        results = validate_dataset(args)

        if args.output == "human":
            print_human_output(results, args)
        else:
            print_json_output(results)

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
