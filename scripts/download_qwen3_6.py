"""Download a Qwen3.6-35B-A3B model for REAP.

Usage:
    python scripts/download_qwen3_6.py [--model MODEL_NAME]

Defaults to ``Qwen/Qwen3.6-35B-A3B``.
"""

import os
import argparse
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(
        description="Download Qwen3.6-35B-A3B model for REAP"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.6-35B-A3B",
        help="HuggingFace model repo id (default: Qwen/Qwen3.6-35B-A3B)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.normpath(
        os.path.join(script_dir, os.pardir, "artifacts", "models")
    )
    model_dir = os.path.join(artifacts_dir, args.model.split("/")[-1])

    print(f"Downloading {args.model} → {model_dir}")
    snapshot_download(
        repo_id=args.model,
        repo_type="model",
        local_dir=model_dir,
    )
    print(f"Model saved to {model_dir}")
    print()
    print("To run REAP pruning:")
    print(
        f"  bash experiments/pruning-cli.sh 0 {args.model} reap 42 0.25 "
        "\"theblackcat102/evol-codealpaca-v1:4096\" true true true false false"
    )


if __name__ == "__main__":
    main()
