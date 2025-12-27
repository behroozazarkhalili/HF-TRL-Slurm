"""
Typer CLI for SLURM job generation.

Usage:
    python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7
    python -m generator generate Qwen/Qwen2.5-0.5B HuggingFaceH4/ultrachat_200k -m sft -p 0.5 -o jobs/my-job.sh
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm

from .smart_defaults import get_defaults, get_job_name, get_hub_model_id

app = typer.Typer(
    name="generator",
    help="SLURM Job Generator for HF-TRL Training",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    model: str = typer.Argument(..., help="HuggingFace model ID (e.g., Qwen/Qwen2.5-7B)"),
    dataset: str = typer.Argument(..., help="HuggingFace dataset ID"),
    method: str = typer.Option(..., "-m", "--method", help="Training method: sft/grpo/dpo"),
    params_b: float = typer.Option(..., "-p", "--params", help="Model size in billions (e.g., 7 for 7B)"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output path for job script"),
    streaming: bool = typer.Option(False, "--streaming", help="Use streaming for large datasets"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", help="Max samples for streaming"),
    reward_type: str = typer.Option("combined", "--reward-type", help="Reward type for GRPO"),
    max_length: int = typer.Option(2048, "--max-length", help="Maximum sequence/completion length"),
    max_prompt_length: int = typer.Option(512, "--max-prompt-length", help="Maximum prompt length (GRPO)"),
    username: str = typer.Option("ermiaazarkhalili", "-u", "--username", help="HuggingFace username"),
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation and generate directly"),
):
    """Generate a SLURM job script for training.

    ALWAYS shows the resolved configuration first, then asks for confirmation.
    Use --yes to skip confirmation.

    Examples:
        # GRPO for 7B model
        python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7

        # SFT with streaming
        python -m generator generate Qwen/Qwen2.5-0.5B nvidia/OpenMathInstruct-2 -m sft -p 0.5 --streaming

        # Save to file
        python -m generator generate Qwen/Qwen2.5-7B nvidia/OpenMathInstruct-2 -m grpo -p 7 -o jobs/my-job.sh
    """
    from .job_generator import JobGenerator

    # Validate method
    if method not in ['sft', 'grpo', 'dpo']:
        console.print(f"[red]Error: Invalid method '{method}'. Use sft, grpo, or dpo.[/red]")
        raise typer.Exit(1)

    # Get smart defaults based on model size
    config = get_defaults(
        model_id=model,
        params_b=params_b,
        method=method,
        streaming=streaming,
        max_samples=max_samples,
        reward_type=reward_type,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
    )

    # Add derived values
    config['job_name'] = get_job_name(model, dataset, method)
    config['hub_model_id'] = get_hub_model_id(username, model, dataset, method)
    config['gguf_repo_id'] = f"{config['hub_model_id']}-GGUF"
    config['dataset_id'] = dataset

    # =========================================
    # ALWAYS show configuration first (dry-run)
    # =========================================
    console.print("\n[bold cyan]═══ Job Configuration Preview ═══[/bold cyan]\n")

    # Model info table
    model_table = Table(title="Model", show_header=False, box=None)
    model_table.add_column("Key", style="cyan", width=20)
    model_table.add_column("Value", style="green")
    model_table.add_row("Model ID", model)
    model_table.add_row("Size", f"{params_b}B ({config['size_category']})")
    model_table.add_row("4-bit Quantization", str(config['requires_4bit']))
    console.print(model_table)

    # Dataset info table
    console.print()
    dataset_table = Table(title="Dataset", show_header=False, box=None)
    dataset_table.add_column("Key", style="cyan", width=20)
    dataset_table.add_column("Value", style="green")
    dataset_table.add_row("Dataset ID", dataset)
    dataset_table.add_row("Streaming", str(config.get('streaming', False)))
    if config.get('streaming'):
        dataset_table.add_row("Max Samples", str(config.get('max_samples', 'N/A')))
    console.print(dataset_table)

    # Training config table
    console.print()
    training_table = Table(title=f"Training ({method.upper()})", show_header=False, box=None)
    training_table.add_column("Key", style="cyan", width=20)
    training_table.add_column("Value", style="green")
    training_table.add_row("Batch Size", str(config['batch_size']))
    training_table.add_row("Gradient Accumulation", str(config['grad_accum']))
    training_table.add_row("Effective Batch Size", str(config['batch_size'] * config['grad_accum']))
    training_table.add_row("Learning Rate", str(config['lr']))
    training_table.add_row("LoRA Rank", str(config['lora_r']))
    training_table.add_row("LoRA Alpha", str(config['lora_alpha']))
    training_table.add_row("Max Length", str(config['max_length']))
    training_table.add_row("Time Limit", config['time_limit'])
    if method == 'grpo':
        training_table.add_row("Num Generations", str(config.get('num_gen', 4)))
        training_table.add_row("Reward Type", config.get('reward_type', 'combined'))
        training_table.add_row("Max Prompt Length", str(config['max_prompt_length']))
    console.print(training_table)

    # Hardware config table
    console.print()
    hw_table = Table(title="SLURM Configuration", show_header=False, box=None)
    hw_table.add_column("Key", style="cyan", width=20)
    hw_table.add_column("Value", style="green")
    hw_table.add_row("Job Name", config['job_name'])
    hw_table.add_row("Account", config['account'])
    hw_table.add_row("Partition", config['partition'])
    hw_table.add_row("GPU", config['gres'])
    hw_table.add_row("Memory", config['memory'])
    hw_table.add_row("CPUs", str(config['cpus']))
    console.print(hw_table)

    # Output info table
    console.print()
    output_table = Table(title="Output", show_header=False, box=None)
    output_table.add_column("Key", style="cyan", width=20)
    output_table.add_column("Value", style="green")
    output_table.add_row("Hub Model ID", config['hub_model_id'])
    output_table.add_row("GGUF Repo ID", config['gguf_repo_id'])
    console.print(output_table)

    console.print("\n[bold cyan]═══════════════════════════════════[/bold cyan]\n")

    # =========================================
    # Ask for confirmation (unless --yes)
    # =========================================
    if not yes:
        if not Confirm.ask("[yellow]Generate job script with this configuration?[/yellow]"):
            console.print("[red]Cancelled.[/red]")
            raise typer.Exit()

    # Generate job script
    generator = JobGenerator()
    script = generator.generate(
        model_id=model,
        dataset_id=dataset,
        method=method,
        config=config,
    )

    # Output
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(script)
        console.print(f"\n[green]✓ Generated:[/green] {output}")
        console.print(f"\n[dim]Submit with: sbatch {output}[/dim]")
    else:
        console.print("\n[bold]Generated Script:[/bold]\n")
        console.print(script)


@app.command()
def show_defaults(
    method: str = typer.Option("sft", "-m", "--method", help="Training method: sft/grpo/dpo"),
):
    """Show default configurations for each model size category."""
    from .smart_defaults import SIZE_DEFAULTS

    console.print(f"\n[bold cyan]═══ {method.upper()} Defaults by Model Size ═══[/bold cyan]\n")

    for size in ['small', 'medium', 'large']:
        size_config = SIZE_DEFAULTS[size]
        method_config = size_config[method]

        table = Table(title=f"{size.upper()} (≤{1.5 if size == 'small' else 4 if size == 'medium' else 14}B)")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Batch Size", str(method_config['batch_size']))
        table.add_row("Gradient Accumulation", str(method_config['grad_accum']))
        table.add_row("Effective Batch", str(method_config['batch_size'] * method_config['grad_accum']))
        table.add_row("Learning Rate", str(method_config['lr']))
        table.add_row("LoRA Rank", str(method_config['lora_r']))
        table.add_row("LoRA Alpha", str(method_config['lora_alpha']))
        table.add_row("LoRA Dropout", str(method_config.get('lora_dropout', 0.0)))
        if method == 'grpo' and 'num_gen' in method_config:
            table.add_row("Num Generations", str(method_config['num_gen']))
        table.add_row("Requires 4-bit", str(size_config['requires_4bit']))
        table.add_row("Time Limit", size_config['time_limit'][method])

        console.print(table)
        console.print()


@app.command()
def list_templates():
    """List available job templates."""
    templates_dir = Path(__file__).parent.parent / "templates"

    console.print("\n[bold cyan]═══ Available Templates ═══[/bold cyan]\n")

    table = Table()
    table.add_column("Template", style="cyan")
    table.add_column("Method", style="green")
    table.add_column("Description")

    templates = {
        'sft_pipeline.sh.j2': ('SFT', 'Supervised Fine-Tuning pipeline'),
        'grpo_pipeline.sh.j2': ('GRPO', 'Group Relative Policy Optimization pipeline'),
        'dpo_pipeline.sh.j2': ('DPO', 'Direct Preference Optimization pipeline'),
    }

    for template_name, (method, desc) in templates.items():
        template_path = templates_dir / template_name
        status = "✓" if template_path.exists() else "✗"
        table.add_row(f"{status} {template_name}", method, desc)

    console.print(table)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
