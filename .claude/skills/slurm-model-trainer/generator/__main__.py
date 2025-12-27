"""
Entry point for the generator CLI.

Usage:
    python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7
    python -m generator show-defaults -m grpo
    python -m generator list-templates
"""

from .cli import main

if __name__ == "__main__":
    main()
