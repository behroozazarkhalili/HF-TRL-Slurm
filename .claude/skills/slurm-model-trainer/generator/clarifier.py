"""
Interactive clarification when the generator cannot make confident decisions.

This module handles user prompts when automatic detection is uncertain,
using Rich for beautiful terminal output.
"""

from dataclasses import dataclass
from typing import Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()


@dataclass
class Uncertainty:
    """Represents an uncertain decision that needs user input."""

    category: str  # 'column_mapping', 'split_selection', 'training_method', etc.
    question: str
    options: list[str]
    default: Optional[str] = None
    context: str = ""


class Clarifier:
    """Handles interactive questioning when automatic detection is uncertain."""

    def __init__(self, verbose: bool = True):
        """Initialize clarifier.

        Args:
            verbose: Whether to print context information
        """
        self.verbose = verbose

    def ask_column_mapping(self, columns: list[str], needed_role: str) -> str:
        """Ask user which column maps to a specific role.

        Args:
            columns: Available column names in the dataset
            needed_role: The role we need to map (e.g., 'prompt', 'response', 'messages')

        Returns:
            Selected column name
        """
        console.print(f"\n[yellow]Cannot automatically detect '{needed_role}' column[/yellow]")
        console.print(f"Available columns: [cyan]{', '.join(columns)}[/cyan]")
        return Prompt.ask(
            f"Which column should be used as '{needed_role}'?",
            choices=columns,
        )

    def ask_split_selection(self, available_splits: list[str]) -> str:
        """Ask which dataset split to use.

        Args:
            available_splits: Available split names

        Returns:
            Selected split name
        """
        console.print(f"\n[yellow]Multiple splits available[/yellow]")

        # Show splits in a table
        table = Table(show_header=False, box=None)
        table.add_column("Split", style="cyan")
        for split in available_splits:
            table.add_row(split)
        console.print(table)

        default = "train" if "train" in available_splits else available_splits[0]
        return Prompt.ask(
            "Which split should be used?",
            choices=available_splits,
            default=default,
        )

    def ask_training_method(self, detected_format: Optional[str] = None) -> str:
        """Ask which training method to use.

        Args:
            detected_format: Detected dataset format (if any)

        Returns:
            Training method: 'sft', 'grpo', or 'dpo'
        """
        if detected_format:
            console.print(
                f"\n[yellow]Dataset format '{detected_format}' supports multiple methods[/yellow]"
            )
        else:
            console.print("\n[yellow]Please select a training method[/yellow]")

        # Method recommendations based on format
        method_info = {
            'sft': 'Supervised Fine-Tuning - for instruction/chat datasets',
            'grpo': 'Group Relative Policy Optimization - for math/reasoning with rewards',
            'dpo': 'Direct Preference Optimization - for preference datasets',
        }

        table = Table(title="Training Methods", show_header=True)
        table.add_column("Method", style="cyan")
        table.add_column("Description", style="white")
        for method, desc in method_info.items():
            table.add_row(method.upper(), desc)
        console.print(table)

        # Suggest default based on format
        format_defaults = {
            'conversational': 'sft',
            'instruction': 'sft',
            'preference': 'dpo',
            'math': 'grpo',
            'text': 'sft',
        }
        default = format_defaults.get(detected_format, 'sft')

        return Prompt.ask(
            "Which training method?",
            choices=['sft', 'grpo', 'dpo'],
            default=default,
        )

    def ask_model_variant(self, base_id: str, instruct_id: str, method: str) -> str:
        """Ask whether to use base or instruct model.

        Args:
            base_id: Base model ID
            instruct_id: Instruct variant model ID
            method: Training method

        Returns:
            Selected model ID
        """
        if method == 'grpo':
            console.print(f"\n[yellow]GRPO typically uses instruct models[/yellow]")
            console.print(f"  Base model: [cyan]{base_id}[/cyan]")
            console.print(f"  Instruct model: [cyan]{instruct_id}[/cyan]")

            if Confirm.ask(
                f"Use instruct model ({instruct_id})?",
                default=True,
            ):
                return instruct_id
        return base_id

    def ask_reward_type(self, problem_type: Optional[str] = None) -> str:
        """Ask which reward function to use for GRPO.

        Args:
            problem_type: Detected problem type (e.g., 'math', 'coding')

        Returns:
            Reward type: 'combined', 'accuracy', 'format', 'length'
        """
        console.print("\n[yellow]Select reward type for GRPO training[/yellow]")

        reward_info = {
            'combined': 'Accuracy + Format rewards (recommended for math)',
            'accuracy': 'Only check answer correctness',
            'format': 'Only check response format',
            'length': 'Penalize overly long responses',
        }

        table = Table(title="Reward Types", show_header=True)
        table.add_column("Type", style="cyan")
        table.add_column("Description", style="white")
        for rtype, desc in reward_info.items():
            table.add_row(rtype, desc)
        console.print(table)

        # Default based on problem type
        default = 'combined' if problem_type == 'math' else 'accuracy'

        return Prompt.ask(
            "Which reward type?",
            choices=list(reward_info.keys()),
            default=default,
        )

    def confirm_streaming(self, num_rows: Optional[int] = None) -> bool:
        """Confirm streaming mode for large datasets.

        Args:
            num_rows: Number of rows in dataset (if known)

        Returns:
            Whether to use streaming mode
        """
        if num_rows:
            console.print(f"\n[yellow]Dataset has {num_rows:,} rows[/yellow]")
            default = num_rows > 100_000
        else:
            console.print("\n[yellow]Dataset size unknown[/yellow]")
            default = False

        return Confirm.ask(
            "Use streaming mode? (recommended for large datasets)",
            default=default,
        )

    def ask_max_samples(self, default: int = 500_000) -> int:
        """Ask for maximum samples when using streaming.

        Args:
            default: Default max samples

        Returns:
            Maximum number of samples to use
        """
        console.print("\n[yellow]Streaming mode enabled[/yellow]")
        response = Prompt.ask(
            "Maximum samples to use",
            default=str(default),
        )
        return int(response)

    def confirm_4bit_quantization(self, model_size: str, requires_4bit: bool) -> bool:
        """Confirm 4-bit quantization setting.

        Args:
            model_size: Model size category
            requires_4bit: Whether 4-bit is recommended

        Returns:
            Whether to use 4-bit quantization
        """
        if requires_4bit:
            console.print(
                f"\n[yellow]Large model ({model_size}) - 4-bit quantization recommended[/yellow]"
            )
            return Confirm.ask("Use 4-bit quantization?", default=True)
        return False

    def ask_custom_batch_size(
        self,
        default_batch: int,
        default_grad_accum: int,
    ) -> tuple[int, int]:
        """Ask for custom batch size and gradient accumulation.

        Args:
            default_batch: Default batch size
            default_grad_accum: Default gradient accumulation

        Returns:
            Tuple of (batch_size, gradient_accumulation)
        """
        console.print("\n[yellow]Customize batch settings?[/yellow]")
        console.print(
            f"Current: batch_size={default_batch}, grad_accum={default_grad_accum} "
            f"(effective={default_batch * default_grad_accum})"
        )

        if not Confirm.ask("Modify batch settings?", default=False):
            return default_batch, default_grad_accum

        batch = int(Prompt.ask("Batch size", default=str(default_batch)))
        grad_accum = int(Prompt.ask("Gradient accumulation", default=str(default_grad_accum)))

        console.print(f"[green]Effective batch size: {batch * grad_accum}[/green]")
        return batch, grad_accum

    def confirm_configuration(self, config: dict) -> bool:
        """Show configuration and ask for confirmation.

        Args:
            config: Full configuration dictionary

        Returns:
            Whether user confirms the configuration
        """
        console.print("\n[bold cyan]═══ Configuration Summary ═══[/bold cyan]\n")

        # Model info
        model_table = Table(title="Model", show_header=False, box=None)
        model_table.add_column("Key", style="cyan")
        model_table.add_column("Value", style="green")
        model_table.add_row("Model ID", config.get('model_id', 'N/A'))
        model_table.add_row("Size", f"{config.get('params_b', '?')}B ({config.get('size_category', '?')})")
        model_table.add_row("4-bit Quantization", str(config.get('requires_4bit', False)))
        console.print(model_table)

        # Training info
        console.print()
        training_table = Table(title="Training", show_header=False, box=None)
        training_table.add_column("Key", style="cyan")
        training_table.add_column("Value", style="green")
        training_table.add_row("Batch Size", str(config.get('batch_size', 'N/A')))
        training_table.add_row("Gradient Accumulation", str(config.get('grad_accum', 'N/A')))
        effective = config.get('batch_size', 1) * config.get('grad_accum', 1)
        training_table.add_row("Effective Batch Size", str(effective))
        training_table.add_row("Learning Rate", str(config.get('lr', 'N/A')))
        training_table.add_row("LoRA Rank", str(config.get('lora_r', 'N/A')))
        training_table.add_row("LoRA Alpha", str(config.get('lora_alpha', 'N/A')))
        training_table.add_row("Time Limit", config.get('time_limit', 'N/A'))
        if 'num_gen' in config:
            training_table.add_row("Num Generations", str(config['num_gen']))
        if 'reward_type' in config:
            training_table.add_row("Reward Type", config['reward_type'])
        console.print(training_table)

        # Hardware info
        console.print()
        hw_table = Table(title="Hardware", show_header=False, box=None)
        hw_table.add_column("Key", style="cyan")
        hw_table.add_column("Value", style="green")
        hw_table.add_row("Partition", config.get('partition', 'N/A'))
        hw_table.add_row("Memory", config.get('memory', 'N/A'))
        hw_table.add_row("CPUs", str(config.get('cpus', 'N/A')))
        console.print(hw_table)

        console.print("\n[bold cyan]═════════════════════════════[/bold cyan]\n")

        return Confirm.ask("[yellow]Generate job script with this configuration?[/yellow]")
