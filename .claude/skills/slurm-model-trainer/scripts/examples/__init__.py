"""
HuggingFace-Style Example Scripts.

These are minimal, self-contained training scripts designed to be:
- Easy to understand (< 150 lines each)
- Copy-paste ready for customization
- Runnable with `uv run` or embedded in SBATCH scripts

Example Scripts:
- train_sft_example.py: Supervised Fine-Tuning
- train_dpo_example.py: Direct Preference Optimization
- train_grpo_example.py: Group Relative Policy Optimization

Usage:
    # With uv (auto-installs dependencies from PEP 723 header)
    uv run train_sft_example.py

    # Or embed in SBATCH script
    python /path/to/train_sft_example.py

Customize the constants at the top of each script for your use case.
"""
