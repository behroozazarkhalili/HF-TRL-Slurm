"""
Environment Setup Builder for SLURM job scripts.

Generates the environment configuration section including:
- Shell variable exports
- Module loads
- Virtual environment activation
- HuggingFace cache and token setup
"""

from .base import BaseBuilder


class EnvironmentBuilder(BaseBuilder):
    """Build the environment setup section of a SLURM job script.

    This builder generates:
    1. Job information echo statements
    2. Module loads (gcc, arrow, python)
    3. Virtual environment activation
    4. Environment variable exports (HF_HOME, caches, etc.)
    5. HF token loading from .env file
    6. Configuration summary echo statements

    Example output:
        echo "=========================================="
        echo "Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)"
        ...
        module load gcc arrow python/3.11.5
        source /scratch/ermia/venvs/hf_env/bin/activate
        ...
    """

    def build(self) -> str:
        """Build the complete environment setup section.

        Returns:
            The environment setup script section.
        """
        sections = [
            self._config_variables(),
            self._job_info_banner(),
            self._module_loads(),
            self._virtualenv_activation(),
            self._environment_variables(),
            self._hf_token_setup(),
            self._output_directory_setup(),
            self._config_echo_statements(),
        ]
        return "\n".join(sections)

    def _config_variables(self) -> str:
        """Generate configuration shell variables.

        Returns:
            Shell variable declarations for the job configuration.
        """
        c = self.config

        # Base configuration variables (all methods)
        variables = f"""{self._comment_block("Configuration")}
MODEL_NAME='{c.model_id}'
DATASET_NAME='{c.dataset_id}'

# Sample size configuration (for model naming when using streaming)
MAX_SAMPLES={c.max_samples if c.max_samples else 0}
SAMPLE_SIZE_LABEL='{c.sample_size_label}'

HUB_MODEL_ID='{c.hub_model_id}'
GGUF_REPO_ID='{c.gguf_repo_id}'

# Training parameters ({c.size_category} model config)
BATCH_SIZE={c.batch_size}
GRAD_ACCUM={c.grad_accum}
LEARNING_RATE={c.lr}
NUM_EPOCHS={c.num_train_epochs}
LORA_R={c.lora_r}
LORA_ALPHA={c.lora_alpha}"""

        # Method-specific variables
        if self.is_grpo:
            variables += f"""
MAX_COMPLETION_LENGTH={c.max_length}
MAX_PROMPT_LENGTH={c.max_prompt_length}
NUM_GENERATIONS={c.num_gen}
REWARD_TYPE='{c.reward_type}'"""
        else:
            # SFT and DPO use MAX_SEQ_LENGTH
            variables += f"""
MAX_SEQ_LENGTH={c.max_length}"""

        return variables + "\n"

    def _job_info_banner(self) -> str:
        """Generate job information echo statements.

        Returns:
            Echo statements showing job information.
        """
        return f"""
{self._comment_block("Environment Setup")}
{self._echo_banner(f"Job: $SLURM_JOB_NAME (ID: $SLURM_JOB_ID)")}
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo ""

mkdir -p logs
"""

    def _module_loads(self) -> str:
        """Generate module load commands.

        Returns:
            Module load commands for the cluster.
        """
        return """# Load modules
module load gcc arrow python/3.11.5
"""

    def _virtualenv_activation(self) -> str:
        """Generate virtual environment activation.

        Returns:
            Source command to activate the virtualenv.
        """
        return f"""# Activate virtual environment
source {self.venv_path}/bin/activate
"""

    def _environment_variables(self) -> str:
        """Generate environment variable exports.

        Returns:
            Export statements for cache and output directories.
        """
        return f"""# Set environment variables
export SCRATCH=${{SCRATCH:-/scratch/$USER}}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export OUTPUT_DIR="$SCRATCH/outputs/{self.config.job_name}-$SLURM_JOB_ID"
"""

    def _hf_token_setup(self) -> str:
        """Generate HuggingFace token loading.

        Returns:
            Commands to load HF token from .env file.
        """
        return f"""# Load HF token
if [[ -f "{self.env_file_path}" ]]; then
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        # Remove surrounding quotes from value
        value="${{value%\\"}}"; value="${{value#\\"}}"
        value="${{value%\\'}}"; value="${{value#\\'}}"
        export "$key=$value"
    done < "{self.env_file_path}"
fi
"""

    def _output_directory_setup(self) -> str:
        """Generate output directory creation.

        Returns:
            mkdir command for the output directory.
        """
        return """mkdir -p $OUTPUT_DIR
"""

    def _config_echo_statements(self) -> str:
        """Generate configuration summary echo statements.

        Returns:
            Echo statements summarizing the job configuration.
        """
        echo_statements = '''echo ""
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Dataset: $DATASET_NAME"
echo "  Hub Model ID: $HUB_MODEL_ID"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM))"'''

        # Add GRPO-specific echoes
        if self.is_grpo:
            echo_statements += '''
echo "  Num Generations: $NUM_GENERATIONS"
echo "  Reward Type: $REWARD_TYPE"'''

        echo_statements += '''
echo "  LoRA Rank: $LORA_R"
echo "  Output Dir: $OUTPUT_DIR"
echo ""
'''
        return echo_statements
