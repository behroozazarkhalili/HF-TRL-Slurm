"""
Smart defaults based on model size and training method.

Model size categories:
- small: <= 1.5B params (0.5B, 0.6B, 1.5B, 1.7B)
- medium: 1.5B - 4B params (3B, 4B)
- large: 4B - 14B params (7B, 8B, 14B)
"""

from typing import Optional


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
    'mig_40gb': {
        'gres': 'gpu:nvidia_h100_80gb_hbm3_3g.40gb:1',
        'memory': '64G',
        'cpus': 8,
    },
    'mig_10gb': {
        'gres': 'gpu:nvidia_h100_80gb_hbm3_1g.10gb:1',
        'memory': '32G',
        'cpus': 4,
    },
}

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
    'account': 'def-maxwl_gpu',
}


def categorize_model_size(params_b: float) -> str:
    """Categorize model by parameter count.

    Args:
        params_b: Model size in billions of parameters

    Returns:
        Size category: 'small', 'medium', or 'large'
    """
    if params_b <= 1.5:
        return 'small'
    elif params_b <= 4:
        return 'medium'
    return 'large'


def get_partition(method: str, size_category: str) -> str:
    """Get the appropriate SLURM partition based on training method and size.

    Args:
        method: Training method (sft, grpo, dpo)
        size_category: Model size category

    Returns:
        Partition name
    """
    # GRPO and large models need longer partitions
    if method == 'grpo' or size_category == 'large':
        return 'gpubase_bygpu_b5'  # Allows up to 7 days
    return 'gpubase_bygpu_b3'  # Standard partition


def get_defaults(
    model_id: str,
    params_b: float,
    method: str,
    streaming: bool = False,
    max_samples: Optional[int] = None,
    reward_type: str = 'combined',
    max_length: int = 2048,
    max_prompt_length: int = 512,
) -> dict:
    """Get smart defaults for a model/method combination.

    Args:
        model_id: HuggingFace model ID
        params_b: Model size in billions
        method: Training method (sft, grpo, dpo)
        streaming: Whether to use streaming for large datasets
        max_samples: Maximum samples for streaming (default: 500000)
        reward_type: Reward type for GRPO (default: combined)
        max_length: Maximum sequence/completion length
        max_prompt_length: Maximum prompt length (GRPO only)

    Returns:
        Dictionary with all configuration values
    """
    size = categorize_model_size(params_b)
    method_defaults = SIZE_DEFAULTS[size][method].copy()

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

        # Hardware
        'gres': HARDWARE_PROFILES['mig_40gb']['gres'],
        'memory': HARDWARE_PROFILES['mig_40gb']['memory'],
        'cpus': HARDWARE_PROFILES['mig_40gb']['cpus'],
        'partition': get_partition(method, size),

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


def get_job_name(model_id: str, dataset_id: str, method: str) -> str:
    """Generate a job name from model and dataset.

    Args:
        model_id: HuggingFace model ID
        dataset_id: HuggingFace dataset ID
        method: Training method

    Returns:
        SLURM job name
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


def get_hub_model_id(username: str, model_id: str, dataset_id: str, method: str) -> str:
    """Generate Hub model ID for the trained model.

    Args:
        username: HuggingFace username
        model_id: Base model ID
        dataset_id: Dataset ID
        method: Training method

    Returns:
        Hub model ID for the trained model
    """
    model_name = model_id.split('/')[-1]
    dataset_name = dataset_id.split('/')[-1].replace('OpenMathInstruct-2', 'OpenMath2')
    dataset_name = dataset_name.replace('ultrachat_200k', 'UltraChat')

    return f"{username}/{model_name}-{method.upper()}-{dataset_name}"
