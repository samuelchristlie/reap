"""Patches for Qwen3.5/3.6 MoE models (qwen3_5_moe architecture) to support REAP.

Two patches are applied at runtime (no file-level patching required):

1. ``Qwen3_5MoeSparseMoeBlock.forward()`` is replaced to return
   ``(expert_output, router_logits)`` instead of just ``expert_output``,
   so REAP's observer can extract router logits via ``*_, router_logits = output``.

2. ``Qwen3_5MoeExperts`` is wrapped with an iterable adapter so the observer
   can loop over individual experts (via the non-fused path) to compute
   per-expert activations for the saliency metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Expert wrapper – makes a single fused-param expert callable like an
# nn.Module with a standard ``forward(x) -> x`` interface.
# ---------------------------------------------------------------------------


class _Qwen3_5MoeExpertWrapper(nn.Module):
    """Wraps one expert's fused parameter slices so it is callable.

    Forward pass::

        gate, up = chunk(Linear(x, gate_up_proj), 2)
        output   = Linear(act(gate) * up, down_proj)
    """

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, act_fn):
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = F.linear(x, self.gate_up_proj).chunk(2, dim=-1)
        return F.linear(self.act_fn(gate) * up, self.down_proj)


# ---------------------------------------------------------------------------
# Iterable adapter – wraps a ``Qwen3_5MoeExperts`` so that ``enumerate``
    # and indexing work, while preserving the original ``forward()``.
# ---------------------------------------------------------------------------


class Qwen3_5MoeExpertsIterable(nn.Module):
    """Wraps a ``Qwen3_5MoeExperts`` to support ``len``, indexing, and iteration.

    The original ``forward(hidden_states, top_k_index, top_k_weights)`` is
    preserved for normal model execution.  The adapter additionally provides
    per-expert access so that REAP's observer can compute activations for every
    expert individually.
    """

    def __init__(self, experts_module: nn.Module):
        super().__init__()
        self._original = experts_module
        # Re-register the fused parameters on this wrapper so that pruning
        # code can access them via ``moe.experts.gate_up_proj`` etc.
        # Because nn.Parameter is reference-counted, modifying ``.data``
        # on one reference is visible through the other.
        self.gate_up_proj = experts_module.gate_up_proj
        self.down_proj = experts_module.down_proj
        self.num_experts = experts_module.num_experts
        self.hidden_dim = experts_module.hidden_dim
        self.intermediate_dim = experts_module.intermediate_dim
        self.act_fn = experts_module.act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Delegate to the original experts module."""
        return self._original(hidden_states, top_k_index, top_k_weights)

    # -- Sequence-like interface for the observer --------------------------

    def __len__(self) -> int:
        return self.num_experts

    def __getitem__(self, idx: int) -> _Qwen3_5MoeExpertWrapper:
        if idx < 0:
            idx += self.num_experts
        if not (0 <= idx < self.num_experts):
            raise IndexError(
                f"Expert index {idx} out of range [0, {self.num_experts})"
            )
        return _Qwen3_5MoeExpertWrapper(
            gate_up_proj=self.gate_up_proj[idx],
            down_proj=self.down_proj[idx],
            act_fn=self.act_fn,
        )

    def __iter__(self):
        for i in range(self.num_experts):
            yield self[i]


# ---------------------------------------------------------------------------
# SparseMoeBlock patch
# ---------------------------------------------------------------------------


def _patch_sparse_moe_block(module: nn.Module):
    """Patch a single ``Qwen3_5MoeSparseMoeBlock`` to return *router_logits*."""
    # Wrap experts so the observer can iterate over them
    module.experts = Qwen3_5MoeExpertsIterable(module.experts)

    # Capture submodules by value for the closure
    shared_expert = module.shared_expert
    gate = module.gate
    experts = module.experts
    shared_expert_gate = module.shared_expert_gate

    def patched_forward(hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = shared_expert(hidden_states_reshaped)
        router_logits, routing_weights, selected_experts = gate(
            hidden_states_reshaped
        )
        expert_output = experts(
            hidden_states_reshaped, selected_experts, routing_weights
        )

        shared_expert_output = (
            F.sigmoid(shared_expert_gate(hidden_states_reshaped))
            * shared_expert_output
        )

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output, router_logits

    module.forward = patched_forward


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def patch_qwen3_5_moe_model(model: nn.Module) -> nn.Module:
    """Apply all REAP compatibility patches to a Qwen3.5/3.6 MoE model.

    Patches every ``Qwen3_5MoeSparseMoeBlock`` in *model* so that:

    1. ``forward`` returns ``(expert_output, router_logits)``.
    2. The experts module supports per-expert iteration.

    Returns the same *model* reference (patches are applied in-place).
    """
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )

    patched_count = 0
    for _name, module in model.named_modules():
        if isinstance(module, Qwen3_5MoeSparseMoeBlock):
            _patch_sparse_moe_block(module)
            patched_count += 1

    if patched_count == 0:
        raise RuntimeError(
            "patch_qwen3_5_moe_model: found no Qwen3_5MoeSparseMoeBlock "
            "modules in the model. Check that the model uses the "
            "qwen3_5_moe architecture."
        )

    logger.info(
        "Patched %d Qwen3_5MoeSparseMoeBlock modules to return router_logits",
        patched_count,
    )
    return model
