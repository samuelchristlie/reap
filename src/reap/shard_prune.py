"""Shard-level pruning for MoE models.

Prunes experts by operating directly on safetensors shard files using
memory-mapped I/O.  Only the expert and router weight tensors for the
current shard are loaded into memory, so the full model is **never**
resident in RAM.

Typical peak memory per shard: the shard size (~3–4 GB) plus sliced
expert tensors (~1.5 GB per layer).  Well within 32 GB RAM.

Supports both fused experts (Qwen3.5/3.6, Llama-4) where weights are
stored as 3-D ``nn.Parameter`` tensors, and non-fused experts (Qwen3,
Mixtral, …) where each expert is a separate ``nn.Linear`` module.

Usage
-----
After calibrating with a quantised model (layerwise or standard)::

    python -m reap.shard_prune \\
        --model_dir   artifacts/models/Qwen3.6-35B-A3B \\
        --observer_data artifacts/.../observer_data.pkl \\
        --prune_method reap \\
        --compression_ratio 0.25 \\
        --output_dir  artifacts/pruned/Qwen3.6-35B-A3B-reap-0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from reap.model_util import MODEL_ATTRS, get_super_expert_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_model_class(config_path: Path) -> str:
    """Return the first architecture name from a model's ``config.json``."""
    with open(config_path) as f:
        config = json.load(f)
    architectures = config.get("architectures", [])
    if not architectures:
        raise ValueError(f"No 'architectures' found in {config_path}")
    return architectures[0]


def _parse_layer_idx(key: str, layers_path: str) -> Optional[int]:
    """Extract the decoder-layer index from a safetensors key.

    ``"model.language_model.layers.3.mlp.experts.gate_up_proj"`` → ``3``
    """
    pattern = rf"{re.escape(layers_path)}\.(\d+)\."
    m = re.search(pattern, key)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Determine which experts to keep
# ---------------------------------------------------------------------------

def determine_retained_experts(
    observer_data: Dict[int, dict],
    prune_method: str,
    compression_ratio: float,
    preserve_super_experts: bool = False,
    preserve_outliers: bool = False,
) -> Dict[int, List[int]]:
    """Return *per-layer* list of retained expert indices.

    Mirrors the pruning-decision logic in ``prune.py`` without touching
    any model weights.
    """
    # Expert probabilities (used internally and for super-expert logic)
    for layer in observer_data:
        if "expert_proba" not in observer_data[layer]:
            observer_data[layer]["expert_proba"] = (
                observer_data[layer]["expert_frequency"]
                / observer_data[layer]["total_tokens"]
            )

    saliency_key = "expert_frequency" if prune_method == "frequency" else prune_method

    # Optionally protect super experts by setting their saliency to +inf
    if preserve_super_experts or preserve_outliers:
        super_expert_idx = get_super_expert_indices(
            observer_data, include_last_layers=preserve_outliers,
        )
        metrics = [
            "expert_proba", "ean_sum", "ean_mean",
            "weighted_expert_frequency_sum", "weighted_ean_sum",
            "reap", "reap_l2", "weighted_ean_sum_l2",
        ]
        for layer in observer_data:
            sel = super_expert_idx[super_expert_idx[:, 0] == layer][:, 1]
            if len(sel) > 0:
                for metric in metrics:
                    if metric in observer_data[layer]:
                        observer_data[layer][metric][sel] = float("inf")

    retained: Dict[int, List[int]] = {}
    for layer in observer_data:
        num_experts = observer_data[layer]["expert_frequency"].shape[0]
        n_to_prune = int(num_experts * compression_ratio)

        if prune_method == "ean_ca":
            ean = torch.zeros(num_experts)
            for i in range(num_experts):
                ean[i] = torch.linalg.norm(
                    observer_data[layer]["routed_characteristic_activation"][i],
                    dim=-1,
                ).sum()
            _, to_prune = torch.topk(ean, n_to_prune, largest=False)
        else:
            saliency = observer_data[layer].get(saliency_key)
            if saliency is None:
                raise ValueError(
                    f"Prune method '{prune_method}' not found in observer data "
                    f"for layer {layer}.  Available: {list(observer_data[layer].keys())}"
                )
            _, to_prune = torch.topk(saliency, n_to_prune, largest=False)

        retained[layer] = sorted(
            i for i in range(num_experts) if i not in to_prune.tolist()
        )

    return retained


# ---------------------------------------------------------------------------
# Shard-level pruning core
# ---------------------------------------------------------------------------

def prune_model_shards(
    model_dir: Path,
    output_dir: Path,
    retained_experts: Dict[int, List[int]],
    model_class_name: str,
):
    """Prune experts by operating directly on safetensors shard files.

    For each shard:
    1. Open with ``safe_open`` (memory-mapped, no full load).
    2. For expert / router tensors belonging to a pruned layer, load,
       slice by retained indices, and store the result.
    3. For all other tensors, copy through unchanged.
    4. Write the new shard to *output_dir*.

    Peak RAM ≈ one shard (3–4 GB) + sliced expert tensors (~1.5 GB per
    layer in that shard).
    """
    attrs = MODEL_ATTRS[model_class_name]
    layers_path = attrs.get("layers_path", "model.layers")
    moe_block = attrs["moe_block"]
    experts_name = attrs["experts"]
    router_name = attrs["router"]
    fused = attrs["fused"]

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_files = sorted(model_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    sample_layer = next(iter(retained_experts))
    num_retained = len(retained_experts[sample_layer])
    # Original expert count = highest retained index + 1
    num_total = max(
        max(indices) for indices in retained_experts.values()
    ) + 1

    logger.info("Architecture : %s (%s)", model_class_name,
                "fused" if fused else "non-fused")
    logger.info("Layers path  : %s", layers_path)
    logger.info("Experts      : %d → %d  (%.0f%% pruned)",
                num_total, num_retained,
                (1 - num_retained / num_total) * 100)
    logger.info("Shards       : %d", len(shard_files))

    total_size = 0
    weight_map: Dict[str, str] = {}

    for shard_path in shard_files:
        logger.info("Processing %s …", shard_path.name)
        new_tensors: Dict[str, torch.Tensor] = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                layer_idx = _parse_layer_idx(key, layers_path)
                modified = False

                if layer_idx is not None and layer_idx in retained_experts:
                    indices = retained_experts[layer_idx]
                    prefix = f"{layers_path}.{layer_idx}.{moe_block}."
                    router_key = f"{prefix}{router_name}.weight"

                    if fused:
                        # --- Fused experts (Qwen3.5/3.6, Llama-4) ---
                        expert_prefix = f"{prefix}{experts_name}."
                        if key.startswith(expert_prefix):
                            proj = key[len(expert_prefix):]
                            if proj in ("gate_up_proj", "down_proj"):
                                tensor = f.get_tensor(key)
                                new_tensors[key] = tensor[indices]
                                del tensor
                                modified = True
                                logger.debug("  sliced %s", key)

                        if not modified and key == router_key:
                            tensor = f.get_tensor(key)
                            new_tensors[key] = tensor[indices]
                            del tensor
                            modified = True
                            logger.debug("  sliced %s", key)

                    else:
                        # --- Non-fused experts (Qwen3, Mixtral, …) ---
                        expert_prefix = f"{prefix}{experts_name}."
                        old_to_new = {
                            old: new for new, old in enumerate(indices)
                        }
                        if key.startswith(expert_prefix):
                            suffix = key[len(expert_prefix):]
                            m = re.match(r"(\d+)\.(.+)", suffix)
                            if m:
                                old_idx = int(m.group(1))
                                rest = m.group(2)
                                if old_idx in old_to_new:
                                    new_idx = old_to_new[old_idx]
                                    new_key = f"{expert_prefix}{new_idx}.{rest}"
                                    new_tensors[new_key] = f.get_tensor(key)
                                # else: pruned expert → drop tensor
                                modified = True

                        if not modified and key == router_key:
                            tensor = f.get_tensor(key)
                            new_tensors[key] = tensor[indices]
                            del tensor
                            modified = True

                # Pass-through: copy tensor unchanged
                if not modified:
                    new_tensors[key] = f.get_tensor(key)

        # --- Write new shard ---
        out_path = output_dir / shard_path.name
        save_file(new_tensors, str(out_path))

        # Track metadata for the index file
        for k, t in new_tensors.items():
            weight_map[k] = shard_path.name
            total_size += t.nelement() * t.element_size()

        del new_tensors
        logger.info("  → wrote %s", out_path.name)

    # --- Copy non-weight files (tokenizer, generation_config, etc.) ---
    for item in model_dir.iterdir():
        if item.is_file():
            if item.suffix == ".safetensors":
                continue
            if item.name in ("config.json", "model.safetensors.index.json"):
                continue
            shutil.copy2(item, output_dir / item.name)
    for item in model_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)

    # --- Write updated config.json ---
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    ne_key = attrs["num_experts"]
    config[ne_key] = num_retained
    if "text_config" in config and ne_key in config.get("text_config", {}):
        config["text_config"][ne_key] = num_retained

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info("config.json updated: %s = %d", ne_key, num_retained)

    # --- Write updated safetensors index ---
    index_src = model_dir / "model.safetensors.index.json"
    if index_src.exists():
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)
        logger.info("model.safetensors.index.json updated")

    logger.info("Pruned model saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prune MoE experts from safetensors shards without loading "
            "the full model into memory."
        ),
    )
    parser.add_argument(
        "--model_dir", type=Path, required=True,
        help="Path to the original (e.g. BF16) model directory.",
    )
    parser.add_argument(
        "--observer_data", type=Path, required=True,
        help="Path to the observer-data .pkl from calibration.",
    )
    parser.add_argument(
        "--prune_method", default="reap",
        help="Saliency criterion (reap, frequency, ean_ca, …). Default: reap",
    )
    parser.add_argument(
        "--compression_ratio", type=float, default=0.25,
        help="Fraction of experts to prune per layer (default: 0.25).",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output directory for the pruned model.",
    )
    parser.add_argument(
        "--preserve_super_experts", action="store_true",
        help="Never prune identified super experts.",
    )
    parser.add_argument(
        "--preserve_outliers", action="store_true",
        help="Preserve outlier experts across all layers.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.model_dir.is_dir():
        parser.error(f"model_dir does not exist: {args.model_dir}")
    if not args.observer_data.is_file():
        parser.error(f"observer_data not found: {args.observer_data}")
    if args.output_dir.exists() and not args.overwrite:
        if any(args.output_dir.glob("*.safetensors")):
            parser.error(
                f"Output already contains safetensors: {args.output_dir}. "
                "Use --overwrite to replace."
            )

    # --- Detect architecture ---
    model_class = _detect_model_class(args.model_dir / "config.json")
    if model_class not in MODEL_ATTRS:
        parser.error(
            f"Unsupported architecture '{model_class}'. "
            f"Supported: {list(MODEL_ATTRS.keys())}"
        )
    logger.info("Detected architecture: %s", model_class)

    # --- Load observer data ---
    logger.info("Loading observer data from %s …", args.observer_data)
    observer_data = torch.load(args.observer_data, weights_only=False)
    first_layer = next(iter(observer_data))
    logger.info(
        "Observer data: %d layers, %d experts/layer",
        len(observer_data),
        observer_data[first_layer]["expert_frequency"].shape[0],
    )

    # --- Determine pruning ---
    retained = determine_retained_experts(
        observer_data,
        args.prune_method,
        args.compression_ratio,
        args.preserve_super_experts,
        args.preserve_outliers,
    )
    num_retained = len(retained[first_layer])
    num_total = observer_data[first_layer]["expert_frequency"].shape[0]
    logger.info(
        "Pruning %d/%d experts per layer (%.0f%%)",
        num_total - num_retained, num_total,
        args.compression_ratio * 100,
    )

    # --- Prune ---
    prune_model_shards(
        args.model_dir,
        args.output_dir,
        retained,
        model_class,
    )


if __name__ == "__main__":
    main()
