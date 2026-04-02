"""
Training Block Builder for SLURM job scripts.

Generates the training command section including:
- Phase header
- Python training script invocation
- Method-specific arguments (SFT, DPO, GRPO)
- Error handling
"""

from .base import BaseBuilder


class TrainingBlockBuilder(BaseBuilder):
    """Build the training command section of a SLURM job script.

    This builder generates:
    1. Phase header banner
    2. Python command with all training arguments
    3. Exit code checking and error handling

    The builder automatically selects the correct:
    - Training script (train_sft.py, train_dpo.py, train_grpo.py)
    - Method-specific arguments
    - Sequence length variable name

    Example output:
        # =============================================================================
        # Phase 1: Training
        # =============================================================================
        echo "=========================================="
        echo "Phase 1: GRPO Training"
        echo "=========================================="

        python /path/to/scripts/train_grpo.py \\
            --model_name_or_path $MODEL_NAME \\
            ...
    """

    def build(self) -> str:
        """Build the complete training command section.

        Returns:
            The training phase script section.
        """
        sections = [
            self._phase_header(),
            self._training_command(),
            self._error_handling(),
        ]
        return "\n".join(sections)

    def _phase_header(self) -> str:
        """Generate the phase 1 header.

        Returns:
            Header banner for the training phase.
        """
        return f"""{self._comment_block("Phase 1: Training")}
{self._echo_banner(f"Phase 1: {self.method_upper} Training")}
"""

    def _training_command(self) -> str:
        """Generate the complete Python training command.

        Returns:
            The Python command with all arguments.
        """
        script_name = f"train_{self.config.method}.py"
        script_path = f"{self.skill_path}/scripts/{script_name}"

        # Build argument list
        args = [
            f"python {script_path} \\",
            "    --model_name_or_path $MODEL_NAME \\",
            "    --dataset_name $DATASET_NAME \\",
            "    --output_dir $OUTPUT_DIR \\",
        ]

        # Add common training arguments
        args.extend(self._common_args())

        # Add method-specific arguments
        args.extend(self._method_specific_args())

        return "\n".join(args)

    def _common_args(self) -> list[str]:
        """Generate common training arguments shared across all methods.

        Returns:
            List of argument strings (each ending with backslash).
        """
        c = self.config

        args = [
            "    --num_train_epochs $NUM_EPOCHS \\",
            "    --per_device_train_batch_size $BATCH_SIZE \\",
            "    --gradient_accumulation_steps $GRAD_ACCUM \\",
            "    --learning_rate $LEARNING_RATE \\",
        ]

        # 4-bit quantization flag
        if c.requires_4bit:
            args.append("    --use_4bit \\")

        # Precision and memory optimization
        args.extend([
            "    --bf16 \\",
            "    --gradient_checkpointing \\",
        ])

        # LoRA configuration
        args.extend([
            "    --lora_r $LORA_R \\",
            "    --lora_alpha $LORA_ALPHA \\",
            f"    --lora_dropout {c.lora_dropout} \\",
        ])

        # Checkpointing configuration
        args.extend([
            "    --save_strategy steps \\",
            f"    --save_steps {c.save_steps} \\",
            f"    --save_total_limit {c.save_total_limit} \\",
            f"    --logging_steps {c.logging_steps} \\",
        ])

        # Hub push configuration
        args.extend([
            "    --push_to_hub \\",
            "    --hub_model_id $HUB_MODEL_ID \\",
            f"    --hub_strategy {c.hub_strategy} \\",
        ])

        # Tracking/logging configuration
        project_name = c.job_name.rsplit("-", 1)[0]
        args.extend([
            f"    --report_to {c.report_to} \\",
            f'    --project "{project_name}" \\',
            f'    --run_name "{c.job_name}-$SLURM_JOB_ID" \\',
        ])

        # Streaming configuration
        if c.streaming:
            args.extend([
                "    --streaming \\",
                f"    --max_samples {c.max_samples} \\",
            ])

        return args

    def _method_specific_args(self) -> list[str]:
        """Generate method-specific training arguments.

        Returns:
            List of method-specific argument strings.
        """
        if self.is_sft:
            return self._sft_args()
        elif self.is_grpo:
            return self._grpo_args()
        elif self.is_dpo:
            return self._dpo_args()
        else:
            raise ValueError(f"Unknown training method: {self.config.method}")

    def _sft_args(self) -> list[str]:
        """Generate SFT-specific arguments.

        Returns:
            SFT argument strings.
        """
        return [
            "    --max_length $MAX_SEQ_LENGTH \\",
            "    --per_device_eval_batch_size $BATCH_SIZE",
            "",  # Empty line for readability
        ]

    def _grpo_args(self) -> list[str]:
        """Generate GRPO-specific arguments.

        Returns:
            GRPO argument strings.
        """
        return [
            "    --max_completion_length $MAX_COMPLETION_LENGTH \\",
            "    --max_prompt_length $MAX_PROMPT_LENGTH \\",
            "    --num_generations $NUM_GENERATIONS \\",
            "    --reward_type $REWARD_TYPE",
            "",  # Empty line for readability
        ]

    def _dpo_args(self) -> list[str]:
        """Generate DPO-specific arguments.

        Returns:
            DPO argument strings.
        """
        return [
            "    --max_length $MAX_SEQ_LENGTH",
            "",  # Empty line for readability
        ]

    def _error_handling(self) -> str:
        """Generate exit code checking and error handling.

        Returns:
            Bash code for error handling.
        """
        return """TRAIN_EXIT_CODE=$?

if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed successfully!"
"""
