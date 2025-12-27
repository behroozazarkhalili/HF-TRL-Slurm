#!/usr/bin/env python3
"""
Job Chaining Utility
====================
Create dependent job chains for automated pipelines.

Usage:
    python chain_jobs.py \
        --train_job_id 12345 \
        --eval_tasks comprehensive \
        --convert_gguf
"""

import os
import argparse
import subprocess
from typing import List, Optional


def parse_args():
    parser = argparse.ArgumentParser(description="Chain Slurm jobs with dependencies")

    parser.add_argument("--train_job_id", type=str, required=True,
                        help="Training job ID to depend on")
    parser.add_argument("--model", type=str, required=True,
                        help="Model HF Hub ID (from training)")
    parser.add_argument("--eval_tasks", type=str, default="comprehensive",
                        help="Evaluation task suite")
    parser.add_argument("--convert_gguf", action="store_true",
                        help="Add GGUF conversion job")
    parser.add_argument("--account", type=str, default="def-ACCOUNT_NAME",
                        help="Slurm account")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")

    return parser.parse_args()


def submit_job(
    script: str,
    dependency: Optional[str] = None,
    account: str = "def-ACCOUNT_NAME",
    exports: dict = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Submit a Slurm job and return job ID."""

    cmd = ["sbatch", "--account", account, "--parsable"]

    if dependency:
        cmd.extend(["--dependency", f"afterok:{dependency}"])

    if exports:
        export_str = ",".join(f"{k}={v}" for k, v in exports.items())
        cmd.extend(["--export", f"ALL,{export_str}"])

    cmd.append(script)

    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return "DRY_RUN_JOB_ID"

    print(f"Submitting: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        print(f"Submitted job: {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"stderr: {e.stderr}")
        return None


def main():
    args = parse_args()

    # Get skill directory
    skill_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    templates_dir = os.path.join(skill_dir, "templates")

    print("=" * 60)
    print("Job Chaining Utility")
    print("=" * 60)
    print(f"Training Job: {args.train_job_id}")
    print(f"Model: {args.model}")
    print("")

    jobs = [("Training", args.train_job_id)]

    # Submit evaluation job
    print("\n--- Submitting Evaluation Job ---")
    eval_job_id = submit_job(
        script=os.path.join(templates_dir, "sbatch_eval.sh"),
        dependency=args.train_job_id,
        account=args.account,
        exports={
            "MODEL": args.model,
            "TASKS": args.eval_tasks,
            "PUSH_TO_HUB": "true",
        },
        dry_run=args.dry_run,
    )

    if eval_job_id:
        jobs.append(("Evaluation", eval_job_id))

    # Submit GGUF conversion if requested
    if args.convert_gguf and eval_job_id:
        print("\n--- Submitting GGUF Conversion Job ---")
        gguf_job_id = submit_job(
            script=os.path.join(templates_dir, "sbatch_convert.sh"),
            dependency=eval_job_id,
            account=args.account,
            exports={
                "MODEL": args.model,
                "OUTPUT_REPO": f"{args.model}-gguf",
            },
            dry_run=args.dry_run,
        )

        if gguf_job_id:
            jobs.append(("GGUF Conversion", gguf_job_id))

    # Print summary
    print("\n" + "=" * 60)
    print("Job Chain Summary")
    print("=" * 60)
    for i, (name, job_id) in enumerate(jobs):
        arrow = "→" if i < len(jobs) - 1 else ""
        print(f"  {i+1}. {name}: {job_id} {arrow}")

    print("")
    print("Monitor with:")
    print(f"  squeue -u $USER")
    print(f"  sacct -j {','.join(j[1] for j in jobs)}")


if __name__ == "__main__":
    main()
