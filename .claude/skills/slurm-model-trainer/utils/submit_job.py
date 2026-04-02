#!/usr/bin/env python3
"""
Job submission helper for SLURM.

Provides a wrapper around sbatch with validation, job ID extraction,
and status checking functionality.

Usage:
    python submit_job.py jobs/my-training-job.sh
    python submit_job.py jobs/my-training-job.sh --dry-run
    python submit_job.py jobs/my-training-job.sh --wait
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


def validate_job_script(script_path: Path) -> Tuple[bool, list]:
    """Validate a SLURM job script before submission.

    Args:
        script_path: Path to the job script.

    Returns:
        Tuple of (is_valid, list of issues).
    """
    issues = []

    if not script_path.exists():
        issues.append(f"Script file does not exist: {script_path}")
        return False, issues

    content = script_path.read_text()
    lines = content.split('\n')

    # Check for shebang
    if not lines[0].startswith('#!'):
        issues.append("Missing shebang (#!/bin/bash) at start of script")

    # Check for required SBATCH directives
    required_directives = ['--job-name', '--account', '--time']
    found_directives = set()

    for line in lines:
        if line.strip().startswith('#SBATCH'):
            for directive in required_directives:
                if directive in line:
                    found_directives.add(directive)

    for directive in required_directives:
        if directive not in found_directives:
            issues.append(f"Missing required SBATCH directive: {directive}")

    # Check for common issues
    if '--gres=gpu' not in content and 'gpu' not in content.lower():
        issues.append("Warning: No GPU resource requested (--gres=gpu)")

    # Check for environment setup
    if 'module load' not in content:
        issues.append("Warning: No 'module load' commands found")

    if 'source' not in content and 'activate' not in content.lower():
        issues.append("Warning: No virtual environment activation found")

    # Check script is executable
    if not os.access(script_path, os.X_OK):
        issues.append("Warning: Script is not executable (chmod +x recommended)")

    is_valid = not any(issue for issue in issues if not issue.startswith('Warning'))
    return is_valid, issues


def submit_job(script_path: Path, dry_run: bool = False) -> Optional[int]:
    """Submit a job using sbatch.

    Args:
        script_path: Path to the job script.
        dry_run: If True, validate but don't submit.

    Returns:
        Job ID if successful, None otherwise.
    """
    # Validate first
    is_valid, issues = validate_job_script(script_path)

    if issues:
        print("Validation results:")
        for issue in issues:
            prefix = "[WARN]" if issue.startswith("Warning") else "[ERROR]"
            print(f"  {prefix} {issue}")
        print()

    if not is_valid:
        print("Cannot submit job due to validation errors.")
        return None

    if dry_run:
        print(f"[DRY RUN] Would submit: {script_path}")
        return None

    # Submit the job
    try:
        result = subprocess.run(
            ['sbatch', str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )

        # Extract job ID from output (format: "Submitted batch job 12345678")
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            job_id = int(match.group(1))
            print(f"Job submitted successfully!")
            print(f"  Job ID: {job_id}")
            print(f"  Script: {script_path}")
            print(f"\nMonitor with:")
            print(f"  squeue -j {job_id}")
            print(f"  tail -f logs/*-{job_id}.out")
            return job_id
        else:
            print(f"Job submitted but could not extract ID: {result.stdout}")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job: {e.stderr}")
        return None
    except FileNotFoundError:
        print("Error: sbatch command not found. Are you on a SLURM cluster?")
        return None


def get_job_status(job_id: int) -> Optional[str]:
    """Get the status of a SLURM job.

    Args:
        job_id: SLURM job ID.

    Returns:
        Job status string or None if not found.
    """
    try:
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '-h', '-o', '%T'],
            capture_output=True,
            text=True
        )

        status = result.stdout.strip()
        return status if status else None

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def wait_for_job(job_id: int, poll_interval: int = 30) -> bool:
    """Wait for a job to complete.

    Args:
        job_id: SLURM job ID.
        poll_interval: Seconds between status checks.

    Returns:
        True if job completed successfully, False otherwise.
    """
    print(f"Waiting for job {job_id} to complete...")

    while True:
        status = get_job_status(job_id)

        if status is None:
            # Job not in queue - check if completed
            result = subprocess.run(
                ['sacct', '-j', str(job_id), '--format=State', '-n', '-P'],
                capture_output=True,
                text=True
            )
            final_status = result.stdout.strip().split('\n')[0] if result.stdout else 'UNKNOWN'

            if final_status == 'COMPLETED':
                print(f"\nJob {job_id} completed successfully!")
                return True
            else:
                print(f"\nJob {job_id} ended with status: {final_status}")
                return False

        print(f"  Status: {status}", end='\r')
        time.sleep(poll_interval)


def cancel_job(job_id: int) -> bool:
    """Cancel a running SLURM job.

    Args:
        job_id: SLURM job ID.

    Returns:
        True if cancellation was successful.
    """
    try:
        subprocess.run(
            ['scancel', str(job_id)],
            check=True
        )
        print(f"Job {job_id} cancelled.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to cancel job: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Submit and manage SLURM jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Submit a job:
    python submit_job.py jobs/my-job.sh

  Validate without submitting:
    python submit_job.py jobs/my-job.sh --dry-run

  Submit and wait for completion:
    python submit_job.py jobs/my-job.sh --wait

  Check job status:
    python submit_job.py --status 12345678

  Cancel a job:
    python submit_job.py --cancel 12345678
"""
    )

    parser.add_argument('script', nargs='?', type=Path,
                        help='Job script to submit')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate script without submitting')
    parser.add_argument('--wait', action='store_true',
                        help='Wait for job to complete after submission')
    parser.add_argument('--status', type=int, metavar='JOB_ID',
                        help='Check status of a job')
    parser.add_argument('--cancel', type=int, metavar='JOB_ID',
                        help='Cancel a running job')
    parser.add_argument('--poll-interval', type=int, default=30,
                        help='Seconds between status checks when waiting (default: 30)')

    args = parser.parse_args()

    # Handle status check
    if args.status:
        status = get_job_status(args.status)
        if status:
            print(f"Job {args.status}: {status}")
        else:
            print(f"Job {args.status} not found in queue (may have completed)")
        return

    # Handle cancel
    if args.cancel:
        cancel_job(args.cancel)
        return

    # Submit job
    if not args.script:
        parser.print_help()
        sys.exit(1)

    job_id = submit_job(args.script, dry_run=args.dry_run)

    if job_id and args.wait:
        success = wait_for_job(job_id, poll_interval=args.poll_interval)
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
