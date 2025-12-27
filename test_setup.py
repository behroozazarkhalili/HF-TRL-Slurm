#!/usr/bin/env python3
"""
Test script to verify the HF-TRL fine-tuning setup.

Run this script to check if all components are properly installed
and configured.

Usage:
    python test_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all required packages can be imported."""
    print("=" * 60)
    print("Testing Package Imports")
    print("=" * 60)

    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT"),
        ("trl", "TRL"),
        ("bitsandbytes", "BitsAndBytes"),
        ("yaml", "PyYAML"),
        ("dotenv", "python-dotenv"),
    ]

    all_ok = True
    for pkg, name in packages:
        try:
            module = __import__(pkg)
            version = getattr(module, "__version__", "unknown")
            print(f"  ✓ {name}: {version}")
        except ImportError as e:
            print(f"  ✗ {name}: NOT INSTALLED - {e}")
            all_ok = False

    return all_ok


def test_finetune_package():
    """Test the finetune package."""
    print("\n" + "=" * 60)
    print("Testing Fine-tune Package")
    print("=" * 60)

    try:
        from src.finetune import (
            FinetuneConfig,
            create_default_config,
            ModelConfig,
            DatasetConfig,
            TrainingConfig,
            LoraConfig,
        )
        print("  ✓ Config classes imported successfully")

        # Test config creation
        config = create_default_config(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            dataset_name="tatsu-lab/alpaca",
            method="sft"
        )
        print(f"  ✓ Default config created: {config.model.name}")

        # Test validation
        errors = config.validate()
        if errors:
            print(f"  ⚠ Validation warnings: {errors}")
        else:
            print("  ✓ Config validation passed")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_quantization_utils():
    """Test quantization utilities."""
    print("\n" + "=" * 60)
    print("Testing Quantization Utilities")
    print("=" * 60)

    try:
        from src.finetune.utils import (
            check_quantization_support,
            estimate_memory_usage,
            get_bnb_config,
        )
        print("  ✓ Quantization utils imported")

        # Check system support
        support = check_quantization_support()
        print(f"  ✓ CUDA available: {support['cuda_available']}")
        print(f"  ✓ BitsAndBytes: {support['bitsandbytes']}")
        if support['cuda_available']:
            print(f"  ✓ GPU: {support.get('gpu_name', 'N/A')}")
            print(f"  ✓ VRAM: {support.get('gpu_memory_gb', 'N/A')} GB")
            print(f"  ✓ BF16 support: {support.get('bf16_support', False)}")

        # Test memory estimation
        mem = estimate_memory_usage(
            model_params_billions=7,
            quantization="4bit",
            lora_enabled=True
        )
        print(f"  ✓ 7B model 4-bit estimate: {mem['total_memory_gb']} GB")
        print(f"  ✓ Recommended GPU: {mem['recommended_gpu']}")

        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_yaml():
    """Test YAML config save/load."""
    print("\n" + "=" * 60)
    print("Testing Config YAML Operations")
    print("=" * 60)

    try:
        from src.finetune import FinetuneConfig, create_default_config
        import tempfile
        import os

        # Create config
        config = create_default_config(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            dataset_name="tatsu-lab/alpaca",
            method="sft"
        )

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name

        config.to_yaml(temp_path)
        print(f"  ✓ Config saved to: {temp_path}")

        # Load back
        loaded_config = FinetuneConfig.from_yaml(temp_path)
        print(f"  ✓ Config loaded: {loaded_config.model.name}")

        # Verify
        assert loaded_config.model.name == config.model.name
        assert loaded_config.training.method == config.training.method
        print("  ✓ Config roundtrip verified")

        # Cleanup
        os.unlink(temp_path)
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_env_file():
    """Test .env file configuration."""
    print("\n" + "=" * 60)
    print("Testing Environment Configuration")
    print("=" * 60)

    from dotenv import load_dotenv
    import os

    load_dotenv()

    hf_token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if hf_token:
        # Mask token for security
        masked = hf_token[:8] + "..." + hf_token[-4:] if len(hf_token) > 12 else "***"
        print(f"  ✓ HF Token found: {masked}")
    else:
        print("  ⚠ HF Token not found in .env (needed for Hub upload)")

    wandb_key = os.getenv("WANDB_API_KEY")
    if wandb_key:
        print("  ✓ WANDB_API_KEY found")
    else:
        print("  ○ WANDB_API_KEY not set (optional)")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HF-TRL Fine-tuning Setup Test")
    print("=" * 60 + "\n")

    results = []

    results.append(("Package Imports", test_imports()))
    results.append(("Fine-tune Package", test_finetune_package()))
    results.append(("Quantization Utils", test_quantization_utils()))
    results.append(("Config YAML", test_config_yaml()))
    results.append(("Environment", test_env_file()))

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready for fine-tuning.\n")
        print("Next steps:")
        print("  1. Read the usage guide: docs/USAGE_GUIDE.md")
        print("  2. Create a config file: configs/my_training.yaml")
        print("  3. Start training!")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("Run: pip install -r requirements.txt")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
