#!/usr/bin/env python3
# /// script
# dependencies = [
#     "trl>=0.26.2",
#     "peft>=0.18.0",
#     "transformers>=4.57.3",
#     "accelerate>=1.12.0",
#     "datasets>=4.4.2",
#     "bitsandbytes>=0.49.0",
#     "trackio>=0.13.1",
#     "liger-kernel>=0.5.0",
# ]
# ///
"""
TRL GRPO (Group Relative Policy Optimization) Training Script
==============================================================
Production-ready GRPO training for online RL with:
- Prompt-only dataset format
- Custom reward functions for math reasoning
- Streaming mode support for large datasets
- Trackio monitoring
- Hub push for model persistence
- TRL Memory Release Techniques for reduced VRAM usage

GRPO Requirements:
- Base model should be instruction-tuned
- Dataset needs only 'prompt' column
- Reward function or reward model

Memory Optimization Options (from HuggingFace TRL docs):
- --use_liger_kernel: Reduces memory by ~60%, increases throughput by ~20%
- --use_vllm: Uses vLLM for faster generation
- --vllm_enable_sleep_mode: Offloads vLLM to CPU during optimization
- --gradient_checkpointing: Trades compute for memory
- --no_ds3_gather_for_generation: Prevents OOM in DeepSpeed ZeRO-3

See: https://huggingface.co/docs/trl/reducing_memory_usage

Supported Datasets:
- nvidia/OpenMathInstruct-2 (default, 14M samples, streaming)
- open-r1/OpenR1-Math-220k (math reasoning)
- trl-lib/math_shepherd
- Any dataset with 'prompt' or 'problem' column

Usage:
    python train_grpo.py \
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
        --dataset_name nvidia/OpenMathInstruct-2 \
        --streaming --max_samples 500000 --seed 42 \
        --output_dir ./output \
        --use_liger_kernel --gradient_checkpointing \
        --push_to_hub \
        --hub_model_id username/my-grpo-model
"""

import os
import argparse
import re
import traceback
from typing import Optional, List, Dict, Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

# Global variable to store ground truth answers for reward computation
# Keys are raw prompt strings (not hash()) to avoid PYTHONHASHSEED non-determinism
# and chat-template mismatch issues
GROUND_TRUTH_ANSWERS: Dict[str, str] = {}


def _prompt_key(prompt: str) -> str:
    """Create a stable lookup key from a prompt string.

    Strips whitespace and chat template tokens to ensure ground truth stored
    during dataset loading can be found during reward computation, even if
    TRL applies a chat template wrapper to the prompt.

    Handles tokens from multiple model families:
    - Qwen: <|im_start|>, <|im_end|>, <|endoftext|>
    - Llama/Mistral: <s>, </s>, [INST], [/INST], <<SYS>>, <</SYS>>
    - Gemma: <start_of_turn>, <end_of_turn>, <bos>, <eos>
    - ChatML: <|system|>, <|user|>, <|assistant|>
    """
    # Qwen-style: <|...|>
    key = re.sub(r'<\|[^|]*\|>', '', prompt)
    # Llama/Mistral-style: [INST], [/INST], <<SYS>>, <</SYS>>
    key = re.sub(r'\[/?INST\]', '', key)
    key = re.sub(r'<<?/?SYS>?>?', '', key)
    # Gemma-style: <start_of_turn>, <end_of_turn>, <bos>, <eos>
    key = re.sub(r'<(?:start_of_turn|end_of_turn|bos|eos)>', '', key)
    # Generic BOS/EOS: <s>, </s>
    key = re.sub(r'</?s>', '', key)
    # Role labels left by chat templates
    key = re.sub(r'^(system|user|model|assistant)\n', '', key, flags=re.MULTILINE)
    return key.strip()

# Try to import trackio
try:
    import trackio
    TRACKIO_AVAILABLE = True
except ImportError:
    TRACKIO_AVAILABLE = False
    print("Warning: trackio not available, using default logging")


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training with TRL")

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to pretrained model (should be instruction-tuned)")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="The name of the dataset (needs 'prompt' column)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration name")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming mode for large datasets")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (shuffling)")

    # GRPO-specific arguments
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of generations per prompt")
    parser.add_argument("--reward_type", type=str, default="combined",
                        choices=["accuracy", "math", "length", "format", "combined", "custom"],
                        help="Type of reward function. 'combined' uses math+accuracy+length+format together (default)")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model and logs")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum number of training steps")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="Learning rate (very low for GRPO)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Maximum sequence length for generated completions")
    parser.add_argument("--max_prompt_length", type=int, default=256,
                        help="Maximum prompt length")

    # Saving arguments
    parser.add_argument("--save_strategy", type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help="Checkpoint save strategy")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save frequency in steps")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep")

    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Hub model ID")
    parser.add_argument("--hub_strategy", type=str, default="end",
                        help="Hub push strategy (end=push final model only)")

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging frequency in steps")
    parser.add_argument("--report_to", type=str, default="trackio",
                        help="Reporting integration")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Run name for tracking")
    parser.add_argument("--project", type=str, default="grpo-training",
                        help="Project name for tracking")
    parser.add_argument("--trackio_space_id", type=str, default=None,
                        help="HF Space ID for trackio (e.g., 'username/space'). If not set, logs locally.")

    # Performance arguments
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")

    # TRL Memory Release Techniques
    # See: https://huggingface.co/docs/trl/main/en/reducing_memory_usage
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for fast generation (requires pip install trl[vllm])")
    parser.add_argument("--vllm_mode", type=str, default="colocate",
                        choices=["server", "colocate"],
                        help="vLLM mode: 'colocate' shares training GPUs, 'server' uses separate vLLM server")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3,
                        help="GPU memory utilization for vLLM (0.0-1.0, default 0.3)")
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true",
                        help="Enable vLLM sleep mode - offloads vLLM params to CPU during optimization step")
    parser.add_argument("--use_liger_kernel", action="store_true",
                        help="Use Liger Kernel to reduce memory by ~60%% and increase throughput by ~20%%")
    parser.add_argument("--ds3_gather_for_generation", action=argparse.BooleanOptionalAction, default=True,
                        help="Gather model weights for generation in DeepSpeed ZeRO-3 (use --no-ds3_gather_for_generation to disable)")

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")

    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load model and tokenizer with optional quantization."""
    print(f"Loading model: {args.model_name_or_path}")

    # Determine compute dtype (bf16/fp16 already auto-detected and corrected by main())
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32

    # Quantization config
    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif args.use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # device_map="auto" is incompatible with DeepSpeed / multi-GPU accelerate.
    # When accelerate handles distribution, we must NOT set device_map.
    use_device_map = (
        not os.environ.get("ACCELERATE_USE_DEEPSPEED")
        and int(os.environ.get("WORLD_SIZE", "1")) <= 1
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        device_map="auto" if use_device_map else None,
        trust_remote_code=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded successfully")

    return model, tokenizer


def load_and_prepare_dataset(args):
    """Load dataset and ensure prompt format. Supports streaming mode."""
    global GROUND_TRUTH_ANSWERS

    print(f"Loading dataset: {args.dataset_name}")
    print(f"Streaming mode: {args.streaming}")
    print(f"Seed: {args.seed}")

    if args.streaming:
        # Load in streaming mode for large datasets
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
            streaming=True,
        )
        print("Dataset loaded in streaming mode")

        # Shuffle with seed for reproducibility
        dataset = dataset.shuffle(seed=args.seed, buffer_size=10000)
        print(f"Shuffled with seed={args.seed}")

        # Take max_samples if specified
        if args.max_samples is not None:
            dataset = dataset.take(args.max_samples)
            print(f"Taking {args.max_samples} samples")

        # Convert streaming dataset to regular dataset for TRL compatibility
        # This will download the samples as we iterate
        print("Converting streaming dataset to list (this may take a while)...")
        samples = []
        for i, example in enumerate(dataset):
            # Process each example based on dataset format
            if "nvidia/OpenMathInstruct" in args.dataset_name:
                # Handle nvidia/OpenMathInstruct-2 format
                prompt = example.get("problem", "")
                answer = example.get("expected_answer", "")
                if prompt and answer:
                    GROUND_TRUTH_ANSWERS[_prompt_key(prompt)] = answer
            elif "open-r1/OpenR1-Math" in args.dataset_name or "openr1" in args.dataset_name.lower():
                # Handle OpenR1-Math format
                prompt = example.get("problem", "")
                answer = example.get("answer", "")
                if prompt and answer:
                    GROUND_TRUTH_ANSWERS[_prompt_key(prompt)] = answer
            elif "NuminaMath" in args.dataset_name or "numina" in args.dataset_name.lower():
                # Handle AI-MO/NuminaMath-CoT format: answer is in \boxed{} inside 'solution'
                prompt = example.get("problem", "")
                solution = example.get("solution", "")
                if prompt and solution:
                    answer = extract_answer(solution)
                    if answer:
                        GROUND_TRUTH_ANSWERS[_prompt_key(prompt)] = answer
            else:
                # Generic handling — try to extract answer from 'solution' field if present
                prompt = example.get("prompt") or example.get("problem") or example.get("question") or ""
                solution = example.get("solution", "")
                if prompt and solution:
                    answer = extract_answer(solution)
                    if answer:
                        GROUND_TRUTH_ANSWERS[_prompt_key(prompt)] = answer

            # Skip empty/whitespace-only prompts
            if prompt and prompt.strip():
                samples.append({"prompt": prompt})
            else:
                continue

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} samples...")

        print(f"Stored {len(GROUND_TRUTH_ANSWERS)} ground truth answers")

        # Convert to Dataset
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
        print(f"Converted to Dataset with {len(dataset)} samples")

    else:
        # Standard non-streaming loading
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split,
        )

        print(f"Dataset columns: {dataset.column_names}")

        # Handle nvidia/OpenMathInstruct-2 dataset format
        if "nvidia/OpenMathInstruct" in args.dataset_name:
            print("Detected nvidia/OpenMathInstruct-2 dataset format")

            # Store ground truth answers for reward computation
            if "expected_answer" in dataset.column_names:
                for i, example in enumerate(dataset):
                    problem = example.get("problem", "")
                    answer = example.get("expected_answer", "")
                    if problem and answer:
                        GROUND_TRUTH_ANSWERS[_prompt_key(problem)] = answer
                print(f"Stored {len(GROUND_TRUTH_ANSWERS)} ground truth answers")

            # Rename 'problem' to 'prompt'
            if "problem" in dataset.column_names and "prompt" not in dataset.column_names:
                dataset = dataset.rename_column("problem", "prompt")
                print("Renamed 'problem' to 'prompt'")

            # Keep only necessary columns for GRPO
            columns_to_keep = ["prompt"]
            columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
                print(f"Removed columns: {columns_to_remove}")

        # Handle OpenR1-Math-220k dataset format
        elif "open-r1/OpenR1-Math" in args.dataset_name or "openr1" in args.dataset_name.lower():
            print("Detected OpenR1-Math dataset format")

            # Store ground truth answers for reward computation
            if "answer" in dataset.column_names:
                for i, example in enumerate(dataset):
                    problem = example.get("problem", "")
                    answer = example.get("answer", "")
                    if problem and answer:
                        # Use hash of problem as key to avoid memory issues with long strings
                        GROUND_TRUTH_ANSWERS[_prompt_key(problem)] = answer
                print(f"Stored {len(GROUND_TRUTH_ANSWERS)} ground truth answers")

            # Rename 'problem' to 'prompt' if needed
            if "problem" in dataset.column_names and "prompt" not in dataset.column_names:
                dataset = dataset.rename_column("problem", "prompt")
                print("Renamed 'problem' to 'prompt'")

            # Keep only necessary columns for GRPO
            columns_to_keep = ["prompt"]
            columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
                print(f"Removed columns: {columns_to_remove}")

        # Handle AI-MO/NuminaMath-CoT format: answer is in \boxed{} inside 'solution'
        elif "NuminaMath" in args.dataset_name or "numina" in args.dataset_name.lower():
            print("Detected NuminaMath-CoT dataset format")

            # Extract ground truth from \boxed{} in solution field
            if "solution" in dataset.column_names:
                for example in dataset:
                    problem = example.get("problem", "")
                    solution = example.get("solution", "")
                    if problem and solution:
                        answer = extract_answer(solution)
                        if answer:
                            GROUND_TRUTH_ANSWERS[_prompt_key(problem)] = answer
                print(f"Stored {len(GROUND_TRUTH_ANSWERS)} ground truth answers")

            # Rename 'problem' to 'prompt'
            if "problem" in dataset.column_names and "prompt" not in dataset.column_names:
                dataset = dataset.rename_column("problem", "prompt")
                print("Renamed 'problem' to 'prompt'")

            # Keep only prompt column
            columns_to_remove = [col for col in dataset.column_names if col != "prompt"]
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
                print(f"Removed columns: {columns_to_remove}")

        # Ensure 'prompt' column exists for other datasets
        elif "prompt" not in dataset.column_names:
            # Try common mappings
            prompt_columns = ["problem", "instruction", "question", "input", "query"]
            for col in prompt_columns:
                if col in dataset.column_names:
                    dataset = dataset.rename_column(col, "prompt")
                    print(f"Renamed '{col}' to 'prompt'")
                    break

        # Shuffle with seed for reproducibility
        dataset = dataset.shuffle(seed=args.seed)
        print(f"Shuffled dataset with seed={args.seed}")

        # Limit samples
        if args.max_samples is not None:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Final dataset size: {len(dataset)}")
    print(f"Final columns: {dataset.column_names}")

    return dataset


def create_peft_config(args) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from the LAST \\boxed{...} handling nested braces correctly.

    Uses a brace-depth counter instead of regex to handle cases like
    \\boxed{\\frac{3}{4}} or \\boxed{x^{2} + 1}.

    Finds the LAST occurrence because math solutions typically present
    the final answer in the last \\boxed{} (earlier ones may be intermediate steps).
    """
    # Find the LAST occurrence of \boxed{
    idx = text.rfind('\\boxed{')
    if idx == -1:
        idx = text.rfind('\\boxed {')
    if idx == -1:
        return None

    # Find the opening brace
    brace_start = text.find('{', idx)
    if brace_start == -1:
        return None

    depth = 1
    pos = brace_start + 1
    while pos < len(text) and depth > 0:
        if text[pos] == '{':
            depth += 1
        elif text[pos] == '}':
            depth -= 1
        pos += 1

    if depth == 0:
        return text[brace_start + 1:pos - 1].strip()
    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract the final answer from a math solution.

    Looks for common answer formats:
    - \\boxed{answer} (with proper nested brace handling)
    - The answer is X
    - Final answer: X
    - = X (at the end)
    """
    if not text:
        return None

    # Try to find boxed answer first (LaTeX format) — use brace-depth parser
    boxed = _extract_boxed_content(text)
    if boxed:
        return boxed

    # Try common answer phrases
    answer_patterns = [
        r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+([^\n.]+)',
        r'[Ff]inal\s+[Aa]nswer[:\s]+([^\n.]+)',
        r'[Aa]nswer[:\s]+([^\n.]+)',
        r'[Tt]herefore[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)',
        r'[Ss]o[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([^\n.]+)',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # Try to find a number at the very end
    last_line_match = re.search(r'=\s*([+-]?\d+(?:\.\d+)?)\s*$', text)
    if last_line_match:
        return last_line_match.group(1).strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    if not answer:
        return ""

    # Remove whitespace and convert to lowercase
    answer = answer.strip().lower()

    # Convert LaTeX fractions to a/b BEFORE stripping (prevents \frac{5}{9} -> "59")
    # Pattern handles one level of nested braces: \frac{3\sqrt{3}}{2} -> 3\sqrt{3}/2
    answer = re.sub(r'\\frac\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1/\2', answer)

    # Remove common LaTeX formatting that wraps content
    answer = re.sub(r'\\text\s*\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathrm\s*\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\sqrt\s*\{([^}]+)\}', r'sqrt(\1)', answer)

    # Remove remaining LaTeX commands
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = re.sub(r'[{}$]', '', answer)

    # Normalize fractions (plain text a / b -> a/b)
    answer = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', answer)

    # Remove trailing punctuation
    answer = re.sub(r'[.,;:!?]+$', '', answer)

    # Normalize whitespace
    answer = ' '.join(answer.split())

    return answer


def detect_reward_type(dataset_name: str, sample_prompts: List[str]) -> str:
    """Auto-detect the appropriate reward type based on dataset name and sample content.

    Analyzes the dataset name and sample prompts to determine if this is a:
    - Math/reasoning dataset -> 'math' reward
    - Code dataset -> 'code' reward (future)
    - General dataset -> 'accuracy' reward
    """
    dataset_lower = dataset_name.lower()

    # Check dataset name for math-related keywords
    math_keywords = ['math', 'gsm', 'numina', 'openr1', 'hendrycks', 'olympiad',
                     'aime', 'amc', 'competition', 'reasoning', 'arithmetic']
    for keyword in math_keywords:
        if keyword in dataset_lower:
            print(f"Auto-detected math dataset from keyword '{keyword}' in dataset name")
            return "math"

    # Analyze sample prompts for math content
    if sample_prompts:
        math_indicators = 0
        total_checked = min(len(sample_prompts), 10)

        for prompt in sample_prompts[:total_checked]:
            prompt_lower = prompt.lower()
            # Check for mathematical content
            if any(word in prompt_lower for word in ['calculate', 'solve', 'prove', 'find the value',
                                                       'equation', 'expression', 'formula', 'sum',
                                                       'product', 'ratio', 'percentage', 'fraction']):
                math_indicators += 1
            # Check for mathematical notation
            if re.search(r'[=+\-*/^].*\d', prompt) or re.search(r'\d.*[=+\-*/^]', prompt):
                math_indicators += 1
            # Check for LaTeX math notation
            if re.search(r'\\[a-zA-Z]+\{', prompt) or '$' in prompt:
                math_indicators += 1

        # If more than 50% of samples appear to be math, use math reward
        if math_indicators >= total_checked * 0.5:
            print(f"Auto-detected math content from prompt analysis ({math_indicators}/{total_checked} indicators)")
            return "math"

    # Default to accuracy
    print("Using default 'accuracy' reward type")
    return "accuracy"


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse a normalized answer string as a number.

    Handles plain floats, simple fractions (3/4), and avoids mangling
    non-numeric strings.
    """
    s = s.strip()
    # Direct float
    try:
        return float(s)
    except ValueError:
        pass
    # Simple fraction like 3/4
    if '/' in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    return None


def _normalize_completions(items: list) -> list:
    """Normalize TRL completions to plain strings.

    TRL passes List[str] for non-conversational datasets or
    List[List[Dict]] for conversational datasets (assistant is first message).
    """
    if not items:
        return []
    first = items[0]
    if isinstance(first, list):
        return [
            i[0]["content"] if i and isinstance(i[0], dict) else str(i)
            for i in items
        ]
    if isinstance(first, str):
        return items
    return [str(i) for i in items]


def _normalize_prompts(items: list) -> list:
    """Normalize TRL prompts to plain strings.

    TRL passes List[str] for non-conversational datasets or
    List[List[Dict]] for conversational datasets (user message is last turn).
    """
    if not items:
        return []
    first = items[0]
    if isinstance(first, list):
        return [
            i[-1]["content"] if i and isinstance(i[-1], dict) else str(i)
            for i in items
        ]
    if isinstance(first, str):
        return items
    return [str(i) for i in items]


def create_reward_function(reward_type: str):
    """Create reward function based on type.

    Reward types:
    - combined: Uses all criteria (math accuracy 50%, format 25%, length 15%, reasoning 10%)
    - math/accuracy: Focus on answer correctness
    - format: Focus on proper formatting (\boxed{}, step-by-step)
    - length: Focus on detailed responses
    """
    global GROUND_TRUTH_ANSWERS

    if reward_type == "combined":
        # Combined reward: math accuracy (50%) + format (25%) + length (15%) + reasoning (10%)
        def reward_fn(completions, prompts, **kwargs):
            completions = _normalize_completions(completions)
            prompts = _normalize_prompts(prompts)
            rewards = []
            for completion, prompt in zip(completions, prompts):
                try:
                    score = 0.0

                    # === MATH ACCURACY (50% weight) ===
                    accuracy_score = 0.0
                    predicted_answer = extract_answer(completion)
                    prompt_hash = _prompt_key(prompt)
                    ground_truth = GROUND_TRUTH_ANSWERS.get(prompt_hash, None)

                    if predicted_answer and ground_truth:
                        pred_norm = normalize_answer(predicted_answer)
                        gt_norm = normalize_answer(ground_truth)

                        if pred_norm == gt_norm:
                            accuracy_score = 1.0  # Exact match
                        else:
                            # Numeric comparison first (prevents "1" in "10" false positive)
                            pred_num = _try_parse_number(pred_norm)
                            gt_num = _try_parse_number(gt_norm)
                            if pred_num is not None and gt_num is not None:
                                if abs(pred_num - gt_num) < 1e-6:
                                    accuracy_score = 1.0  # Numerically equal
                                # else: both numeric but different — 0.0, no partial credit
                            elif pred_norm in gt_norm or gt_norm in pred_norm:
                                accuracy_score = 0.7  # Partial match (non-numeric only)
                    elif predicted_answer:
                        accuracy_score = 0.3  # Has answer but no ground truth

                    # === FORMAT (25% weight) ===
                    format_score = 0.0
                    if re.search(r'\\boxed\{', completion):
                        format_score += 0.5  # Proper boxed answer
                    if re.search(r'(step\s*\d|first|second|third|then|next|finally)', completion.lower()):
                        format_score += 0.3  # Step-by-step reasoning
                    if re.search(r'[=+\-*/^]', completion):
                        format_score += 0.2  # Mathematical notation
                    format_score = min(format_score, 1.0)

                    # === LENGTH (15% weight) ===
                    length_score = 0.0
                    word_count = len(completion.split())
                    if word_count < 10:
                        length_score = 0.0
                    elif word_count < 30:
                        length_score = 0.3
                    elif word_count < 100:
                        length_score = 0.7
                    elif word_count < 300:
                        length_score = 1.0
                    else:
                        length_score = 0.8  # Slight penalty for very long

                    # === REASONING QUALITY (10% weight) ===
                    reasoning_score = 0.0
                    completion_lower = completion.lower()
                    # Check for explanation indicators
                    if any(word in completion_lower for word in ['because', 'therefore', 'since', 'thus', 'hence']):
                        reasoning_score += 0.4
                    # Check for clear logical flow
                    if any(word in completion_lower for word in ['let', 'given', 'we have', 'we get', 'we can']):
                        reasoning_score += 0.3
                    # Check for conclusion
                    if any(word in completion_lower for word in ['answer', 'result', 'solution', 'final']):
                        reasoning_score += 0.3
                    reasoning_score = min(reasoning_score, 1.0)

                    # === WEIGHTED COMBINATION ===
                    score = (accuracy_score * 0.50 +
                             format_score * 0.25 +
                             length_score * 0.15 +
                             reasoning_score * 0.10)

                    rewards.append(score)
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"[CombinedReward] Graceful fallback: {type(e).__name__}: {e}")
                    rewards.append(0.0)
                except Exception as e:
                    traceback.print_exc()
                    print(f"[CombinedReward] UNEXPECTED error — possible bug: {e}")
                    rewards.append(0.0)
            return rewards
        return reward_fn

    elif reward_type == "accuracy" or reward_type == "math":
        # Math accuracy reward - extracts and compares final answers
        def reward_fn(completions, prompts, **kwargs):
            completions = _normalize_completions(completions)
            prompts = _normalize_prompts(prompts)
            rewards = []
            for completion, prompt in zip(completions, prompts):
                try:
                    # Extract answer from completion
                    predicted_answer = extract_answer(completion)

                    # Get ground truth if available
                    prompt_hash = _prompt_key(prompt)
                    ground_truth = GROUND_TRUTH_ANSWERS.get(prompt_hash, None)

                    if predicted_answer and ground_truth:
                        # Compare normalized answers
                        pred_norm = normalize_answer(predicted_answer)
                        gt_norm = normalize_answer(ground_truth)

                        if pred_norm == gt_norm:
                            rewards.append(1.0)  # Exact match
                        else:
                            # Numeric comparison first (prevents "1" in "10" false positive)
                            pred_num = _try_parse_number(pred_norm)
                            gt_num = _try_parse_number(gt_norm)
                            if pred_num is not None and gt_num is not None:
                                if abs(pred_num - gt_num) < 1e-6:
                                    rewards.append(1.0)  # Numerically equal
                                else:
                                    rewards.append(0.0)  # Both numeric but different
                            elif pred_norm in gt_norm or gt_norm in pred_norm:
                                rewards.append(0.7)  # Partial match (non-numeric only)
                            else:
                                rewards.append(0.0)
                    elif predicted_answer:
                        # Has an answer but no ground truth to compare
                        # Check for valid formatting
                        if re.search(r'\\boxed\{', completion):
                            rewards.append(0.5)  # Good format
                        else:
                            rewards.append(0.3)  # Has answer but poor format
                    else:
                        # No answer extracted
                        if len(completion.split()) > 50:
                            rewards.append(0.1)  # At least attempted reasoning
                        else:
                            rewards.append(0.0)
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"[MathReward] Graceful fallback: {type(e).__name__}: {e}")
                    rewards.append(0.0)
                except Exception as e:
                    traceback.print_exc()
                    print(f"[MathReward] UNEXPECTED error — possible bug: {e}")
                    rewards.append(0.0)
            return rewards
        return reward_fn

    elif reward_type == "length":
        # Simple length-based reward (prefer longer, detailed responses)
        def reward_fn(completions, prompts, **kwargs):
            completions = _normalize_completions(completions)
            rewards = []
            for completion in completions:
                length = len(completion.split())
                if length < 10:
                    rewards.append(0.0)
                elif length < 50:
                    rewards.append(0.5)
                elif length < 200:
                    rewards.append(1.0)
                else:
                    rewards.append(0.8)  # Slight penalty for very long
            return rewards
        return reward_fn

    elif reward_type == "format":
        # Reward for proper formatting (boxed answer, step-by-step)
        def reward_fn(completions, prompts, **kwargs):
            completions = _normalize_completions(completions)
            rewards = []
            for completion in completions:
                score = 0.0
                # Check for boxed answer
                if re.search(r'\\boxed\{', completion):
                    score += 0.5
                # Check for step-by-step reasoning
                if re.search(r'(step\s*\d|first|second|third|then|next|finally)', completion.lower()):
                    score += 0.3
                # Check for mathematical notation
                if re.search(r'[=+\-*/^]', completion):
                    score += 0.2
                rewards.append(min(score, 1.0))
            return rewards
        return reward_fn

    else:
        # Default: return neutral reward
        def reward_fn(completions, prompts, **kwargs):
            completions = _normalize_completions(completions)
            return [0.5] * len(completions)
        return reward_fn


def print_gpu_diagnostics():
    """Print GPU environment info for debugging node-specific issues."""
    import os
    print("\n--- GPU Diagnostics ---")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        print(f"Device: {props.name} (compute {props.major}.{props.minor}, {props.total_memory / 1024**3:.1f} GB)")
        print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("ERROR: No GPU detected! Training will fail or run on CPU.")
        print("Check: CUDA_VISIBLE_DEVICES, driver version, MIG configuration.")
    print("---\n")


def main():
    args = parse_args()

    print("=" * 60)
    print("GRPO Training with TRL")
    print("=" * 60)
    print(f"Number of generations: {args.num_generations}")
    print(f"Reward type: {args.reward_type}")

    # Validate push_to_hub requires hub_model_id
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--push_to_hub requires --hub_model_id to be set")

    print_gpu_diagnostics()

    # Auto-detect precision BEFORE loading model: bf16 may not be available on MIG partitions
    if args.bf16:
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            print("WARNING: bf16 requested but not supported on this GPU. Falling back to fp16.")
            args.bf16 = False
            args.fp16 = True

    # Initialize tracking (logs locally by default, set space_id for HF Space)
    if args.report_to == "trackio" and TRACKIO_AVAILABLE:
        run_name = args.run_name or f"grpo-{args.model_name_or_path.split('/')[-1]}"
        trackio_kwargs = {
            "project": args.project,
            "name": run_name,
            "config": {
                "model": args.model_name_or_path,
                "dataset": args.dataset_name,
                "num_generations": args.num_generations,
                "reward_type": args.reward_type,
                "learning_rate": args.learning_rate,
            }
        }
        # Use space_id if explicitly set, otherwise log locally (trackio default)
        if args.trackio_space_id:
            trackio_kwargs["space_id"] = args.trackio_space_id
            print(f"Trackio logging to: HF Space {args.trackio_space_id}")
        else:
            print(f"Trackio logging locally (default)")
        trackio.init(**trackio_kwargs)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Load dataset
    train_dataset = load_and_prepare_dataset(args)

    # Note: 'combined' reward already includes math accuracy at 50% weight,
    # so no auto-downgrade to 'math' — combined is strictly better for math datasets
    # as it also rewards format, length, and reasoning quality.

    # Create LoRA config
    peft_config = create_peft_config(args)

    # Create reward function
    reward_fn = create_reward_function(args.reward_type)

    # GRPO Training arguments
    # See: https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig
    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # GRPO-specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,

        # Training hyperparameters
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        # Precision (args already corrected by auto-detection above)
        bf16=args.bf16,
        fp16=args.fp16,

        # Gradient checkpointing
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False}
            if args.gradient_checkpointing else None,

        # TRL Memory Release Techniques
        # See: https://huggingface.co/docs/trl/reducing_memory_usage
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        use_liger_kernel=args.use_liger_kernel,
        ds3_gather_for_generation=args.ds3_gather_for_generation,

        # Saving
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # Logging
        logging_steps=args.logging_steps,
        logging_first_step=True,
        # Don't pass trackio to Trainer - we handle it manually with local logging
        report_to=[] if args.report_to in ["none", "trackio"] else [args.report_to],
        run_name=args.run_name,

        # Hub
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_strategy=args.hub_strategy,

        # Other
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_fn,
    )

    # Train
    print("\nStarting GRPO training...")
    print("Note: GRPO is online RL - model generates responses during training")
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save final model
    print("\nSaving final model...")
    trainer.save_model()

    # Push to Hub
    if args.push_to_hub:
        print(f"\nPushing to Hub: {args.hub_model_id}")
        trainer.push_to_hub()

    # Log metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Finish tracking
    if args.report_to == "trackio" and TRACKIO_AVAILABLE:
        trackio.finish()

    print("\n" + "=" * 60)
    print("GRPO Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output_dir}")
    if args.push_to_hub:
        print(f"Model pushed to: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
