#!/bin/bash
# =============================================================================
# Environment Setup Script for Slurm Model Trainer
# Fir Cluster (Digital Research Alliance of Canada)
# =============================================================================
# Run this script ONCE on the login node to set up the training environment.
# Usage: source setup_env.sh
# =============================================================================

set -e

echo "=========================================="
echo "Setting up HF-TRL Training Environment"
echo "=========================================="

# Detect project directory
if [[ -z "$PROJECT" ]]; then
    # Try to find project directory from current path
    if [[ "$PWD" =~ /project/[0-9]+/ ]]; then
        export PROJECT=$(echo "$PWD" | grep -oP '/project/[0-9]+/[^/]+')
    else
        echo "ERROR: Cannot detect PROJECT directory. Please set it manually:"
        echo "  export PROJECT=/project/XXXXXX/username"
        exit 1
    fi
fi

echo "Project directory: $PROJECT"

# Set up cache directories
export SCRATCH=${SCRATCH:-/scratch/$USER}
export HF_HOME=$SCRATCH/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TORCH_HOME=$SCRATCH/.cache/torch

echo "Cache directories:"
echo "  HF_HOME: $HF_HOME"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  TORCH_HOME: $TORCH_HOME"

# Create cache directories
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $TORCH_HOME

# Load required modules
echo ""
echo "Loading modules..."
module load python/3.11.5
module load cuda/12.2
module load arrow/17.0.0  # For datasets

# Create virtual environment if it doesn't exist
VENV_DIR=$PROJECT/envs/hf-trl
if [[ ! -d "$VENV_DIR" ]]; then
    echo ""
    echo "Creating virtual environment at $VENV_DIR..."
    python -m venv $VENV_DIR
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install core ML libraries
echo ""
echo "Installing core ML libraries..."
pip install \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    accelerate>=0.27.0 \
    peft>=0.10.0 \
    trl>=0.12.0 \
    bitsandbytes>=0.43.0

# Install Unsloth for faster training
echo ""
echo "Installing Unsloth..."
pip install unsloth

# Install monitoring and evaluation
echo ""
echo "Installing monitoring and evaluation tools..."
pip install \
    trackio>=0.10.0 \
    lm-eval>=0.4.0 \
    evaluate>=0.4.0

# Install utilities
echo ""
echo "Installing utilities..."
pip install \
    huggingface-hub>=0.20.0 \
    python-dotenv>=1.0.0 \
    rich>=13.0.0 \
    typer>=0.9.0 \
    pyyaml>=6.0.0 \
    tqdm>=4.66.0

# Install GGUF conversion dependencies
echo ""
echo "Installing GGUF conversion dependencies..."
pip install \
    sentencepiece>=0.1.99 \
    protobuf>=3.20.0 \
    gguf

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import transformers
import datasets
import accelerate
import peft
import trl
import trackio

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'Accelerate: {accelerate.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'TRL: {trl.__version__}')
print('All packages installed successfully!')
"

# Create .env file template if it doesn't exist
ENV_FILE=$PROJECT/HF-TRL/.env
if [[ ! -f "$ENV_FILE" ]]; then
    echo ""
    echo "Creating .env template..."
    cat > $ENV_FILE << 'EOF'
# Hugging Face Token (required for Hub push)
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: WandB API Key (if using WandB instead of Trackio)
# WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Cache directories (auto-set by setup script)
# HF_HOME=/scratch/$USER/.cache/huggingface
EOF
    echo "Created .env template at $ENV_FILE"
    echo "Please edit this file and add your HF_TOKEN"
fi

# Save environment activation script
ACTIVATE_SCRIPT=$PROJECT/HF-TRL/activate_env.sh
cat > $ACTIVATE_SCRIPT << EOF
#!/bin/bash
# Activate HF-TRL training environment
# Usage: source activate_env.sh

module load python/3.11.5 cuda/12.2 arrow/17.0.0
source $VENV_DIR/bin/activate

export HF_HOME=$HF_HOME
export TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
export TORCH_HOME=$TORCH_HOME

# Load HF token from .env
if [[ -f "$PROJECT/HF-TRL/.env" ]]; then
    export \$(grep -v '^#' $PROJECT/HF-TRL/.env | xargs)
fi

echo "HF-TRL environment activated!"
echo "Python: \$(which python)"
echo "CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())')"
EOF
chmod +x $ACTIVATE_SCRIPT

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in future sessions:"
echo "  source $ACTIVATE_SCRIPT"
echo ""
echo "IMPORTANT: Before submitting jobs, pre-download models/datasets:"
echo "  python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('MODEL_NAME')\""
echo "  python -c \"from datasets import load_dataset; load_dataset('DATASET_NAME')\""
echo ""
echo "Don't forget to add your HF_TOKEN to: $ENV_FILE"
echo ""
