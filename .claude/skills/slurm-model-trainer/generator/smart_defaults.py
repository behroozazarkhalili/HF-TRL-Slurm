"""
Smart defaults based on model size and training method.

Model size categories:
- small: <= 1.5B params (0.5B, 0.6B, 1.5B, 1.7B)
- medium: 1.5B - 4B params (3B, 4B)
- large: 4B - 14B params (7B, 8B, 14B)

Environment Variables (for customization):
- SLURM_ACCOUNT: Override default SLURM account (default: def-maxwl_gpu)
- HF_USERNAME: Default HuggingFace username for model uploads

Smart GPU Selection:
- Uses HuggingFace Hub API to fetch model metadata
- Calculates precise VRAM requirements based on model architecture
- Selects the smallest GPU that fits with adequate headroom
"""

import logging
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, NamedTuple, Optional, TypeAlias

logger = logging.getLogger(__name__)

# Type aliases for clarity
SizeCategory: TypeAlias = Literal["small", "medium", "large"]
TrainingMethod: TypeAlias = Literal["sft", "grpo", "dpo"]
RewardType: TypeAlias = Literal["combined", "math", "accuracy", "format", "length"]
JobConfig: TypeAlias = dict[str, Any]


# =============================================================================
# Model Metadata from HuggingFace Hub
# =============================================================================

class ModelMetadata(NamedTuple):
    """Model architecture metadata from HuggingFace Hub."""

    params_b: float              # Total parameters in billions
    hidden_size: int             # Hidden dimension (d_model)
    num_hidden_layers: int       # Number of transformer layers
    intermediate_size: int       # FFN intermediate dimension
    num_attention_heads: int     # Number of attention heads
    vocab_size: int              # Vocabulary size
    max_position_embeddings: int # Max sequence length the model supports
    from_api: bool               # True if fetched from API, False if estimated


@dataclass
class GPUProfile:
    """GPU hardware profile with availability and cost info."""

    name: str                    # Profile identifier
    vram_gb: int                 # Total VRAM in GB
    usable_vram_gb: float        # Usable VRAM (accounting for overhead)
    gres: str                    # SLURM gres string
    memory: str                  # System memory allocation
    cpus: int                    # CPU cores allocated
    partition_priority: int      # Lower = higher priority (more available)


# Available GPU profiles on Fir cluster (ordered by size)
GPU_PROFILES = {
    'mig_10gb': GPUProfile(
        name='mig_10gb',
        vram_gb=10,
        usable_vram_gb=9.0,      # ~10% overhead
        gres='gpu:nvidia_h100_80gb_hbm3_1g.10gb:1',
        memory='32G',
        cpus=4,
        partition_priority=1,    # Most available (6 MIGs per H100)
    ),
    'mig_20gb': GPUProfile(
        name='mig_20gb',
        vram_gb=20,
        usable_vram_gb=18.0,
        gres='gpu:nvidia_h100_80gb_hbm3_2g.20gb:1',
        memory='32G',
        cpus=8,
        partition_priority=2,    # 3 MIGs per H100
    ),
    'mig_40gb': GPUProfile(
        name='mig_40gb',
        vram_gb=40,
        usable_vram_gb=36.0,
        gres='gpu:nvidia_h100_80gb_hbm3_3g.40gb:1',
        memory='64G',
        cpus=8,
        partition_priority=3,    # 2 MIGs per H100
    ),
    'h100_80gb': GPUProfile(
        name='h100_80gb',
        vram_gb=80,
        usable_vram_gb=72.0,
        gres='gpu:h100:1',
        memory='128G',
        cpus=16,
        partition_priority=4,    # Least available (full node)
    ),
}


@dataclass
class GPURecommendation:
    """GPU selection result with reasoning."""

    profile: GPUProfile
    estimated_vram_gb: float
    headroom_gb: float
    headroom_pct: float
    reason: str
    alternative: Optional[str] = None


# Size-based defaults for each training method
SIZE_DEFAULTS = {
    'small': {  # <= 1.5B
        'sft': {
            'batch_size': 8,
            'grad_accum': 2,
            'lr': 2e-4,
            'lora_r': 32,
            'lora_alpha': 64,
            'lora_dropout': 0.0,
        },
        'grpo': {
            'batch_size': 2,
            'grad_accum': 8,
            'lr': 1e-6,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'num_gen': 4,
        },
        'dpo': {
            'batch_size': 4,
            'grad_accum': 4,
            'lr': 5e-5,
            'lora_r': 32,
            'lora_alpha': 64,
            'lora_dropout': 0.0,
        },
        'requires_4bit': False,
        'time_limit': {
            'sft': '2-00:00:00',
            'grpo': '4-00:00:00',
            'dpo': '2-00:00:00',
        },
    },
    'medium': {  # 1.5B - 4B
        'sft': {
            'batch_size': 4,
            'grad_accum': 4,
            'lr': 2e-4,
            'lora_r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0.0,
        },
        'grpo': {
            'batch_size': 2,
            'grad_accum': 8,
            'lr': 1e-6,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'num_gen': 4,
        },
        'dpo': {
            'batch_size': 2,
            'grad_accum': 8,
            'lr': 5e-5,
            'lora_r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0.0,
        },
        'requires_4bit': False,
        'time_limit': {
            'sft': '2-00:00:00',
            'grpo': '4-00:00:00',
            'dpo': '3-00:00:00',
        },
    },
    'large': {  # 4B - 14B
        'sft': {
            'batch_size': 1,
            'grad_accum': 16,
            'lr': 1e-4,
            'lora_r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0.0,
        },
        'grpo': {
            'batch_size': 1,
            'grad_accum': 16,
            'lr': 5e-7,
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'num_gen': 4,
        },
        'dpo': {
            'batch_size': 1,
            'grad_accum': 16,
            'lr': 2e-5,
            'lora_r': 64,
            'lora_alpha': 128,
            'lora_dropout': 0.0,
        },
        'requires_4bit': True,
        'time_limit': {
            'sft': '4-00:00:00',
            'grpo': '6-00:00:00',
            'dpo': '4-00:00:00',
        },
    },
}

# Hardware profiles for Fir cluster
HARDWARE_PROFILES = {
    'h100_80gb': {
        'gres': 'gpu:h100:1',
        'memory': '128G',
        'cpus': 16,
    },
    'mig_40gb': {
        'gres': 'gpu:nvidia_h100_80gb_hbm3_3g.40gb:1',
        'memory': '64G',
        'cpus': 8,
    },
    'mig_20gb': {
        'gres': 'gpu:nvidia_h100_80gb_hbm3_2g.20gb:1',
        'memory': '32G',
        'cpus': 8,
    },
    'mig_10gb': {
        'gres': 'gpu:nvidia_h100_80gb_hbm3_1g.10gb:1',
        'memory': '32G',
        'cpus': 4,
    },
}

# Partition time limits
PARTITION_TIME_LIMITS = {
    'gpubase_bygpu_b1': '3:00:00',      # 3 hours
    'gpubase_bygpu_b2': '12:00:00',     # 12 hours
    'gpubase_bygpu_b3': '1-00:00:00',   # 1 day
    'gpubase_bygpu_b4': '3-00:00:00',   # 3 days
    'gpubase_bygpu_b5': '7-00:00:00',   # 7 days
    'gpubase_bynode_b1': '3:00:00',     # 3 hours (4x H100)
    'gpubase_bynode_b2': '12:00:00',    # 12 hours (4x H100)
    'gpubase_bynode_b3': '1-00:00:00',  # 1 day (4x H100)
    'gpubase_bynode_b4': '3-00:00:00',  # 3 days (4x H100)
    'gpubase_bynode_b5': '7-00:00:00',  # 7 days (4x H100)
    'gpubackfill': '1-00:00:00',        # 1 day (preemptable)
    'gpupreempt': '122-00:00:00',       # 122 days (will be preempted)
}

# Default SLURM account (override via SLURM_ACCOUNT env var)
DEFAULT_SLURM_ACCOUNT = os.environ.get('SLURM_ACCOUNT', 'def-maxwl_gpu')

# Default HuggingFace username (override via HF_USERNAME env var)
DEFAULT_HF_USERNAME = os.environ.get('HF_USERNAME', 'ermiaazarkhalili')

# Common training settings
COMMON_SETTINGS = {
    'bf16': True,
    'gradient_checkpointing': True,
    'num_train_epochs': 1,
    'save_steps': 500,
    'logging_steps': 10,
    'save_total_limit': 3,
    'hub_strategy': 'end',
    'report_to': 'trackio',
    'account': DEFAULT_SLURM_ACCOUNT,
}

# =============================================================================
# VRAM Estimation Constants (Research-Based)
# =============================================================================
#
# Sources:
# - HuggingFace Transformers: Model Memory Anatomy
#   https://huggingface.co/docs/transformers/main/en/model_memory_anatomy
# - erees.dev: Transformer Memory Arithmetic
#   https://erees.dev/transformer-memory/
# - Modal: How much VRAM for fine-tuning
#   https://modal.com/blog/how-much-vram-need-fine-tuning
# - Oxen.ai: GRPO VRAM Requirements
#   https://ghost.oxen.ai/grpo-vram-requirements-for-the-gpu-poor/
# - TRL GitHub Issue #2709: GRPO memory bottleneck
#   https://github.com/huggingface/trl/issues/2709
#
# Formula breakdown for LoRA + bf16 + gradient checkpointing:
#   Model weights (bf16):     2 bytes/param
#   LoRA adapters:            negligible (~0.1% of model)
#   LoRA optimizer (AdamW):   8 bytes/LoRA_param (negligible total)
#   LoRA gradients:           4 bytes/LoRA_param (negligible total)
#   Activations:              O(layers × hidden × seq × batch)
#   KV Cache (generation):    2 × hidden × layers × seq × 2 bytes
#
# Key insight from TRL #2709: GRPO stores logprobs for ALL generations
# simultaneously before backward pass, multiplying memory by num_generations.

# Typical hidden dimensions by model size (for activation estimation)
# These are approximations for transformer models
HIDDEN_DIM_APPROX = {
    # params_b: hidden_dim
    0.1: 512,    # ~100M models
    0.5: 1024,   # ~500M models (Qwen2.5-0.5B: 896)
    1.0: 2048,   # ~1B models (LFM2-1.2B: 2048)
    1.5: 2048,   # ~1.5B models
    3.0: 3072,   # ~3B models (Qwen2.5-3B: 2048)
    7.0: 4096,   # ~7B models (Llama-7B: 4096)
    14.0: 5120,  # ~14B models (Qwen2.5-14B: 5120)
}

# Typical number of layers by model size
NUM_LAYERS_APPROX = {
    0.1: 12,
    0.5: 24,
    1.0: 24,
    1.5: 28,
    3.0: 36,
    7.0: 32,
    14.0: 40,
}


def _get_hidden_dim(params_b: float) -> int:
    """Estimate hidden dimension from parameter count."""
    # Find closest match
    sizes = sorted(HIDDEN_DIM_APPROX.keys())
    for i, size in enumerate(sizes):
        if params_b <= size:
            return HIDDEN_DIM_APPROX[size]
        if i == len(sizes) - 1:
            return HIDDEN_DIM_APPROX[size]
    return 4096  # Default for very large models


def _get_num_layers(params_b: float) -> int:
    """Estimate number of layers from parameter count."""
    sizes = sorted(NUM_LAYERS_APPROX.keys())
    for i, size in enumerate(sizes):
        if params_b <= size:
            return NUM_LAYERS_APPROX[size]
        if i == len(sizes) - 1:
            return NUM_LAYERS_APPROX[size]
    return 40  # Default for very large models


# =============================================================================
# Smart GPU Selection
# =============================================================================

def fetch_model_params(model_id: str) -> float:
    """Get model parameter count from HuggingFace Hub API.

    Tries the API first, then falls back to regex parsing of the model name.
    Logs warnings when falling back so the user knows the estimate may be wrong.
    """
    # Try API first
    try:
        from huggingface_hub import model_info
        info = model_info(model_id)
        params = info.safetensors.get("total", 0) if info.safetensors else 0
        if params > 0:
            return params / 1e9
    except ImportError:
        logger.warning("huggingface_hub not installed — cannot fetch model metadata")
    except Exception as e:
        logger.warning("Could not fetch model params from Hub for %s: %s", model_id, e)

    # Fallback: parse from name (e.g., "Qwen2.5-0.5B" → 0.5)
    match = re.search(r'(\d+\.?\d*)\s*([bm])', model_id.lower())
    if match:
        val, unit = float(match.group(1)), match.group(2)
        result = val if unit == 'b' else val / 1000
        logger.info("Estimated %s params from model name: %.1fB", model_id, result)
        return result

    logger.warning(
        "Could not determine param count for '%s' — defaulting to 1.0B. "
        "Pass explicit --params to override.", model_id
    )
    return 1.0


def estimate_vram(
    params_b: float,
    method: TrainingMethod,
    *,
    max_length: int = 2048,
    batch_size: int = 1,
    num_generations: int = 4,
) -> float:
    """
    Estimate VRAM in GB using official HuggingFace formulas.

    Formula based on HuggingFace TRL vLLM Memory Estimator:
    https://huggingface.co/spaces/trl-lib/recommend-vllm-memory

    HuggingFace Official Formula Components:
        model_size = params × precision_bytes
        kv_cache_per_token = 2 × num_layers × head_dim × num_kv_heads × precision
        kv_cache_total = kv_cache_per_token × batch_size × seq_len
        buffer = 0.20 × (model_size + kv_cache_total)  # Official 20% buffer
        total = model_size + kv_cache_total + buffer

    Additional sources for training overhead:
        - HuggingFace Model Memory Anatomy (18 bytes/param for mixed precision AdamW)
        - TRL reducing_memory_usage.md (activation offloading, gradient checkpointing)
        - TRL GitHub Issue #2709 (GRPO stores logprobs for all generations)

    Args:
        params_b: Model size in billions of parameters.
        method: Training method ('sft', 'grpo', 'dpo').
        max_length: Max sequence length (SFT) or completion length (GRPO/DPO).
        batch_size: Per-device batch size.
        num_generations: Completions per prompt (GRPO only).

    Returns:
        Estimated VRAM requirement in GB.
    """
    # Get model architecture estimates
    hidden_dim = _get_hidden_dim(params_b)
    num_layers = _get_num_layers(params_b)

    # HuggingFace Official Constants
    HF_PRECISION_BYTES = 2      # bf16
    HF_KV_MULTIPLIER = 2        # K and V tensors
    HF_BUFFER_COEFFICIENT = 0.20  # Official 20% buffer from HF formula

    # =================================================================
    # HuggingFace Official Formula: Model Size
    # model_size = params × precision_bytes
    # =================================================================
    model_size_gb = params_b * HF_PRECISION_BYTES

    # =================================================================
    # HuggingFace Official Formula: KV Cache
    # kv_cache_per_token = 2 × num_layers × head_dim × num_kv_heads × precision
    # Assuming MHA (num_kv_heads = num_attn_heads), head_dim = hidden/heads
    # =================================================================
    num_heads = max(hidden_dim // 128, 1)  # Typical head_dim = 128
    head_dim = hidden_dim / num_heads
    kv_cache_per_token = (
        HF_KV_MULTIPLIER * num_layers * head_dim * num_heads * HF_PRECISION_BYTES
    )
    kv_cache_per_token_gb = kv_cache_per_token / 1e9

    # =================================================================
    # LoRA Training Overhead
    # LoRA trains ~1-2% of params with AdamW (8 bytes/LoRA_param)
    # With gradient checkpointing, activation memory is reduced
    # =================================================================
    lora_overhead_gb = params_b * 0.4  # ~0.4 GB per billion params

    # Activation memory with gradient checkpointing
    # Scales with: batch × seq × hidden × sqrt(layers)
    seq_factor = max_length / 512
    activation_gb = params_b * seq_factor * 0.6 * batch_size

    if method == 'sft':
        # SFT: Standard forward/backward pass
        kv_cache_gb = kv_cache_per_token_gb * batch_size * max_length

        # HuggingFace formula: base + buffer
        base_memory = model_size_gb + kv_cache_gb + lora_overhead_gb + activation_gb
        buffer_gb = HF_BUFFER_COEFFICIENT * base_memory

        return base_memory + buffer_gb

    elif method == 'grpo':
        # =================================================================
        # GRPO: Online RL with generation phase
        #
        # Key insight: GRPO generates num_generations completions per prompt,
        # storing KV cache and logprobs for ALL generations simultaneously.
        #
        # From TRL #2709: "You need all logprobs in memory for all samples"
        # =================================================================

        # KV cache scales with num_generations (parallel completions)
        kv_cache_gb = kv_cache_per_token_gb * batch_size * num_generations * max_length

        # Activation memory scales with num_generations
        grpo_activation_gb = activation_gb * num_generations

        # Logprobs storage for advantage computation
        # vocab_size × seq × batch × num_gen × precision (chunked ~25%)
        vocab_size = 128000
        logprobs_gb = (batch_size * num_generations * max_length * vocab_size * 2) / 1e9 * 0.25

        # HuggingFace formula: base + buffer
        base_memory = (
            model_size_gb +
            kv_cache_gb +
            lora_overhead_gb +
            grpo_activation_gb +
            logprobs_gb
        )
        # GRPO uses 30% buffer (20% HF + 10% for online RL memory growth)
        buffer_gb = 0.30 * base_memory

        return base_memory + buffer_gb

    elif method == 'dpo':
        # DPO: Processes chosen/rejected pairs (2x sequences)
        kv_cache_gb = kv_cache_per_token_gb * batch_size * 2 * max_length
        pair_activation_gb = activation_gb * 2

        # HuggingFace formula: base + buffer
        base_memory = model_size_gb + kv_cache_gb + lora_overhead_gb + pair_activation_gb
        buffer_gb = HF_BUFFER_COEFFICIENT * base_memory

        return base_memory + buffer_gb

    # Fallback: conservative estimate
    return params_b * 5.0 * seq_factor


def select_gpu(
    params_b: float,
    method: TrainingMethod,
    *,
    max_length: int = 2048,
    batch_size: int = 1,
    num_generations: int = 4,
    headroom: float = 1.25,
) -> GPUProfile:
    """
    Select smallest GPU that fits the model with headroom.

    Uses research-based VRAM estimation that accounts for:
    - Model size and architecture
    - Sequence length impact on activations
    - GRPO-specific memory for multiple generations
    - Safety headroom for memory fragmentation

    Args:
        params_b: Model size in billions of parameters.
        method: Training method ('sft', 'grpo', 'dpo').
        max_length: Max sequence/completion length.
        batch_size: Per-device batch size.
        num_generations: Completions per prompt (GRPO only).
        headroom: Safety multiplier (default 1.25 = 25% headroom).

    Returns:
        GPUProfile for the smallest adequate GPU.
    """
    vram_estimate = estimate_vram(
        params_b, method,
        max_length=max_length,
        batch_size=batch_size,
        num_generations=num_generations,
    )
    vram_needed = vram_estimate * headroom

    for name in ['mig_10gb', 'mig_20gb', 'mig_40gb', 'h100_80gb']:
        if GPU_PROFILES[name].usable_vram_gb >= vram_needed:
            return GPU_PROFILES[name]
    return GPU_PROFILES['h100_80gb']


def categorize_model_size(params_b: float) -> SizeCategory:
    """Categorize model by parameter count.

    Args:
        params_b: Model size in billions of parameters.

    Returns:
        Size category: 'small', 'medium', or 'large'.
    """
    if params_b <= 1.5:
        return "small"
    elif params_b <= 4:
        return "medium"
    return "large"


def parse_time_to_hours(time_str: str) -> float:
    """Parse SLURM time format to hours.

    Supports formats: D-HH:MM:SS, HH:MM:SS, D-HH:MM
    """
    if '-' in time_str:
        days, rest = time_str.split('-')
        parts = rest.split(':')
        hours = int(days) * 24 + int(parts[0])
        if len(parts) > 1:
            hours += int(parts[1]) / 60
    else:
        parts = time_str.split(':')
        hours = int(parts[0])
        if len(parts) > 1:
            hours += int(parts[1]) / 60
    return hours


def select_partition(time_limit: str) -> str:
    """Select smallest partition that fits the time limit.

    Fir cluster partitions (gpubase_bygpu_*):
    - b1: 3 hours    - b2: 12 hours   - b3: 1 day
    - b4: 3 days     - b5: 7 days

    Smaller partitions have more availability and faster queue times.
    """
    hours = parse_time_to_hours(time_limit)

    if hours <= 3:
        return "gpubase_bygpu_b1"
    elif hours <= 12:
        return "gpubase_bygpu_b2"
    elif hours <= 24:
        return "gpubase_bygpu_b3"
    elif hours <= 72:
        return "gpubase_bygpu_b4"
    else:
        return "gpubase_bygpu_b5"


def get_defaults(
    model_id: str,
    params_b: float,
    method: TrainingMethod,
    *,
    streaming: bool = False,
    max_samples: Optional[int] = None,
    reward_type: RewardType = "combined",
    max_length: int = 2048,
    max_prompt_length: int = 512,
) -> JobConfig:
    """Get smart defaults for a model/method combination.

    Args:
        model_id: HuggingFace model ID.
        params_b: Model size in billions.
        method: Training method (sft, grpo, dpo).
        streaming: Whether to use streaming for large datasets.
        max_samples: Maximum samples for streaming (default: 500000).
        reward_type: Reward type for GRPO (default: combined).
        max_length: Maximum sequence/completion length.
        max_prompt_length: Maximum prompt length (GRPO only).

    Returns:
        Dictionary with all configuration values.
    """
    size = categorize_model_size(params_b)
    method_defaults = SIZE_DEFAULTS[size][method].copy()

    # Get batch size and num_generations for VRAM estimation
    batch_size = method_defaults.get('batch_size', 1)
    num_generations = method_defaults.get('num_gen', 4) if method == 'grpo' else 1

    # Smart GPU selection with sequence-length awareness
    gpu = select_gpu(
        params_b, method,
        max_length=max_length,
        batch_size=batch_size,
        num_generations=num_generations,
    )
    vram_estimate = estimate_vram(
        params_b, method,
        max_length=max_length,
        batch_size=batch_size,
        num_generations=num_generations,
    )

    # Build configuration
    config = {
        # Model info
        'model_id': model_id,
        'params_b': params_b,
        'size_category': size,

        # Training defaults from size
        **method_defaults,
        'requires_4bit': SIZE_DEFAULTS[size]['requires_4bit'],
        'time_limit': SIZE_DEFAULTS[size]['time_limit'][method],

        # Common settings
        **COMMON_SETTINGS,

        # Hardware (smart GPU selection)
        'gres': gpu.gres,
        'memory': gpu.memory,
        'cpus': gpu.cpus,
        'gpu_profile': gpu.name,
        'gpu_vram': gpu.vram_gb,
        'estimated_vram': round(vram_estimate, 1),
        'partition': select_partition(SIZE_DEFAULTS[size]['time_limit'][method]),

        # Sequence lengths
        'max_length': max_length,
        'max_prompt_length': max_prompt_length,
    }

    # Add streaming settings
    if streaming:
        config['streaming'] = True
        config['max_samples'] = max_samples or 500_000
    else:
        config['streaming'] = False

    # Add GRPO-specific settings
    if method == 'grpo':
        config['reward_type'] = reward_type

    return config


def get_job_name(model_id: str, dataset_id: str, method: TrainingMethod) -> str:
    """Generate a job name from model and dataset.

    Args:
        model_id: HuggingFace model ID.
        dataset_id: HuggingFace dataset ID.
        method: Training method.

    Returns:
        SLURM job name.
    """
    # Extract model name (e.g., "Qwen/Qwen2.5-7B" -> "qwen2.5-7b")
    model_name = model_id.split('/')[-1].lower()

    # Extract dataset name (e.g., "nvidia/OpenMathInstruct-2" -> "openmath2")
    dataset_name = dataset_id.split('/')[-1].lower()
    # Simplify common dataset names
    dataset_name = dataset_name.replace('openmath', 'openmath')
    dataset_name = dataset_name.replace('instruct-2', '2')
    dataset_name = dataset_name.replace('ultrachat_200k', 'ultrachat')

    return f"{model_name}-{method}-{dataset_name}"


def format_sample_size(samples: int) -> str:
    """Convert sample count to human-readable label.

    Args:
        samples: Number of training samples.

    Returns:
        Human-readable label (e.g., 10000 → '10K', 1000000 → '1M').
    """
    if samples >= 1_000_000:
        return f"{samples // 1_000_000}M"
    elif samples >= 1_000:
        return f"{samples // 1_000}K"
    return str(samples)


def get_hub_model_id(
    username: Optional[str],
    model_id: str,
    dataset_id: str,
    method: TrainingMethod,
    sample_size: Optional[int] = None,
) -> str:
    """Generate Hub model ID for the trained model.

    Args:
        username: HuggingFace username (uses HF_USERNAME env var if None).
        model_id: Base model ID.
        dataset_id: Dataset ID.
        method: Training method.
        sample_size: Optional number of training samples (adds suffix like -10K).

    Returns:
        Hub model ID for the trained model.
    """
    if username is None:
        username = DEFAULT_HF_USERNAME

    model_name = model_id.split('/')[-1]
    dataset_name = dataset_id.split('/')[-1].replace('OpenMathInstruct-2', 'OpenMath2')
    dataset_name = dataset_name.replace('ultrachat_200k', 'UltraChat')
    dataset_name = dataset_name.replace('NuminaMath-CoT', 'NuminaMath')

    base_name = f"{username}/{model_name}-{method.upper()}-{dataset_name}"

    if sample_size:
        return f"{base_name}-{format_sample_size(sample_size)}"
    return base_name
