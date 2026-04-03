"""
Post-Processing Builder for SLURM job scripts.

Generates the post-training phases including:
- Model card generation and upload
- Evaluation (GRPO only)
- GGUF conversion
- Summary output
"""

from .base import BaseBuilder


class PostProcessingBuilder(BaseBuilder):
    """Build the post-processing section of a SLURM job script.

    This builder generates:
    1. Model card generation and Hub upload (Phase 2)
    2. Evaluation on math benchmarks (Phase 3, GRPO only)
    3. GGUF conversion (Phase 3 or 4)
    4. Summary with output locations

    The phase numbering adjusts automatically based on training method:
    - SFT/DPO: Model Card (2) -> GGUF (3) -> Summary
    - GRPO: Model Card (2) -> Evaluation (3) -> GGUF (4) -> Summary
    """

    def build(self) -> str:
        """Build the complete post-processing section.

        Returns:
            The post-processing script sections.
        """
        sections = [
            self._model_card_phase(),
        ]

        # GRPO includes evaluation phase
        if self.is_grpo:
            sections.append(self._evaluation_phase())

        sections.extend([
            self._gguf_phase(),
            self._summary_phase(),
        ])

        return "\n".join(sections)

    def _model_card_phase(self) -> str:
        """Generate the model card generation and upload phase.

        Returns:
            Model card generation script section.
        """
        c = self.config

        # Determine which max_length variable to use
        max_length_var = "$MAX_COMPLETION_LENGTH" if self.is_grpo else "$MAX_SEQ_LENGTH"

        return f"""{self._comment_block("Phase 2: Generate Model Card")}
echo ""
{self._echo_banner("Phase 2: Generating Model Card")}

python {self.skill_path}/scripts/generate_model_card.py \\
    --model_name "{self.hub_model_name}" \\
    --base_model "$MODEL_NAME" \\
    --dataset "$DATASET_NAME" \\
    --training_method {self.method_upper} \\
    --author {self.hub_username} \\
    --license cc-by-nc-4.0 \\
    --learning_rate $LEARNING_RATE \\
    --batch_size $BATCH_SIZE \\
    --epochs $NUM_EPOCHS \\
    --max_length {max_length_var} \\
    --lora_r $LORA_R \\
    --lora_alpha $LORA_ALPHA \\
    --hardware "NVIDIA H100 MIG" \\
    --output_dir $OUTPUT_DIR/model_card

# Push model card to Hub
python -c "
import os
from huggingface_hub import HfApi
api = HfApi()
output_dir = os.environ['OUTPUT_DIR']
hub_model_id = os.environ.get('HUB_MODEL_ID', '')
if not hub_model_id:
    print('WARNING: HUB_MODEL_ID not set, skipping model card upload')
else:
    card_path = os.path.join(output_dir, 'model_card', 'README.md')
    if not os.path.exists(card_path):
        print(f'WARNING: Model card not found at {{card_path}}')
    else:
        api.upload_file(
            path_or_fileobj=card_path,
            path_in_repo='README.md',
            repo_id=hub_model_id,
        )
        print(f'Model card uploaded to {{hub_model_id}}')
"
"""

    def _evaluation_phase(self) -> str:
        """Generate the evaluation phase (GRPO only).

        Returns:
            Evaluation script section for math benchmarks.
        """
        return f"""{self._comment_block("Phase 3: Evaluation (Math benchmarks)")}
echo ""
{self._echo_banner("Phase 3: Evaluation (GSM8K, MATH)")}

pip install -q lm-eval

python -c "
import os
import subprocess

model_id = os.environ.get('HUB_MODEL_ID', '')
output_dir = os.path.join(os.environ['OUTPUT_DIR'], 'eval_results')

if not model_id:
    print('WARNING: HUB_MODEL_ID not set, skipping evaluation')
else:
    cmd = [
        'lm_eval',
        '--model', 'hf',
        '--model_args', f'pretrained={{model_id}},trust_remote_code=True',
        '--tasks', 'gsm8k,minerva_math',
        '--batch_size', 'auto',
        '--output_path', output_dir,
        '--log_samples',
    ]

    cmd_str = ' '.join(cmd)
    print(f'Running: {{cmd_str}}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f'Warning: Evaluation had issues: {{result.stderr}}')
"
"""

    def _gguf_phase(self) -> str:
        """Generate the GGUF conversion phase.

        Returns:
            GGUF conversion script section.
        """
        # Phase number depends on whether eval was included
        phase_num = "4" if self.is_grpo else "3"

        return f"""{self._comment_block(f"Phase {phase_num}: GGUF Conversion")}
echo ""
{self._echo_banner(f"Phase {phase_num}: GGUF Conversion (Q4_K_M)")}

python {self.skill_path}/scripts/convert_gguf.py \\
    --model $HUB_MODEL_ID \\
    --base_model $MODEL_NAME \\
    --output_repo $GGUF_REPO_ID \\
    --quantizations "Q4_K_M,Q5_K_M,Q8_0" \\
    --output_dir $OUTPUT_DIR/gguf

GGUF_EXIT_CODE=$?

if [[ $GGUF_EXIT_CODE -ne 0 ]]; then
    echo "WARNING: GGUF conversion had issues (exit code $GGUF_EXIT_CODE)"
else
    echo "GGUF conversion completed successfully!"
fi
"""

    def _summary_phase(self) -> str:
        """Generate the summary output section.

        Returns:
            Summary script section with output locations.
        """
        # Include eval results path for GRPO
        eval_line = '\necho "  Eval Results: $OUTPUT_DIR/eval_results"' if self.is_grpo else ""

        return f"""{self._comment_block("Summary")}
echo ""
{self._echo_banner("Pipeline Complete!")}
echo "End time: $(date)"
echo ""
echo "Outputs:"
echo "  Training Output: $OUTPUT_DIR"
echo "  Model on Hub: https://huggingface.co/$HUB_MODEL_ID"
echo "  GGUF on Hub: https://huggingface.co/$GGUF_REPO_ID"{eval_line}
echo ""
echo "To use with Ollama:"
echo "  ollama pull hf.co/$GGUF_REPO_ID:Q4_K_M"
echo ""
"""
