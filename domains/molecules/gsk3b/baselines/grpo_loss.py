"""Transparent FlowGRPO-style objective for FlowMol3 flow matching.

FlowMol3 mixes continuous coordinates with categorical CTMC modalities and
does not expose exact trajectory log likelihoods.  This adapter therefore
uses negative per-molecule flow-matching loss as a documented log-density
surrogate.  It implements group-relative advantages, a clipped on-policy
ratio, and a fixed-base KL proxy, but must not be described as a literal
reproduction of the continuous-image Flow-GRPO sampler.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from domains.molecules.gsk3b.dpo import copy_dgl_graph, per_graph_flow_loss


def group_relative_advantages(
    rewards: Tensor, group_ids: list[str], epsilon: float = 1e-6
) -> Tensor:
    if rewards.ndim != 1 or len(group_ids) != rewards.numel():
        raise ValueError("GRPO requires one reward and group id per sample")
    advantages = torch.empty_like(rewards)
    for group in sorted(set(group_ids)):
        indices = torch.tensor(
            [i for i, value in enumerate(group_ids) if value == group],
            dtype=torch.long,
        )
        values = rewards[indices]
        if len(indices) < 2:
            raise ValueError("GRPO groups need at least two samples")
        advantages[indices] = (values - values.mean()) / values.std(
            unbiased=False
        ).clamp_min(epsilon)
    return advantages


def clipped_grpo_surrogate(
    policy_logp: Tensor,
    old_logp: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float,
) -> tuple[Tensor, Tensor]:
    if not 0 < clip_epsilon < 1:
        raise ValueError("GRPO clipping epsilon must lie in (0, 1)")
    log_ratio = (policy_logp - old_logp).clamp(-20, 20)
    ratio = log_ratio.exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    return -torch.minimum(unclipped, clipped).mean(), ratio


def flowmol_grpo_step(
    policy: Any,
    old_policy: Any,
    base_reference: Any,
    interpolated: Any,
    times: Tensor,
    *,
    advantages: Tensor,
    clip_epsilon: float,
    kl_beta: float,
    log_density_scale: float,
) -> tuple[Tensor, Mapping[str, Tensor]]:
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    # Some FlowMol vector-field paths attach temporary graph data.  Clone all
    # three inputs before any forward pass so identical policy/old checkpoints
    # are evaluated on byte-equivalent graph state at the on-policy gate.
    policy_graph = copy_dgl_graph(interpolated)
    old_graph = copy_dgl_graph(interpolated)
    base_graph = copy_dgl_graph(interpolated)
    node_batch, _ = get_batch_idxs(policy_graph)
    upper = get_upper_edge_mask(policy_graph)
    policy_output = policy.vector_field(
        policy_graph, times, node_batch_idx=node_batch, upper_edge_mask=upper
    )
    policy_loss = per_graph_flow_loss(
        policy, policy_graph, times, vf_output=policy_output
    )
    with torch.no_grad():
        old_nodes, _ = get_batch_idxs(old_graph)
        old_upper = get_upper_edge_mask(old_graph)
        old_output = old_policy.vector_field(
            old_graph, times, node_batch_idx=old_nodes, upper_edge_mask=old_upper
        )
        old_loss = per_graph_flow_loss(
            old_policy, old_graph, times, vf_output=old_output
        )
        base_nodes, _ = get_batch_idxs(base_graph)
        base_upper = get_upper_edge_mask(base_graph)
        base_output = base_reference.vector_field(
            base_graph, times, node_batch_idx=base_nodes,
            upper_edge_mask=base_upper,
        )
        base_loss = per_graph_flow_loss(
            base_reference, base_graph, times, vf_output=base_output
        )
    policy_logp = -log_density_scale * policy_loss
    old_logp = -log_density_scale * old_loss
    base_logp = -log_density_scale * base_loss
    surrogate, ratio = clipped_grpo_surrogate(
        policy_logp, old_logp, advantages.to(policy_logp),
        clip_epsilon=clip_epsilon,
    )
    kl_proxy = 0.5 * (policy_logp - base_logp).square().mean()
    loss = surrogate + kl_beta * kl_proxy
    return loss, {
        "ratio": ratio.detach(),
        "surrogate_loss": surrogate.detach(),
        "kl_proxy": kl_proxy.detach(),
        "policy_flow_loss": policy_loss.detach(),
        "old_flow_loss": old_loss.detach(),
        "base_flow_loss": base_loss.detach(),
    }


def initial_ratio_diagnostic(
    ratio: Tensor, *, absolute_tolerance: float
) -> tuple[bool, float]:
    """Check theoretical ratio one while allowing bounded floating error."""

    if not 0 < absolute_tolerance <= 1e-4:
        raise ValueError("initial ratio absolute tolerance must lie in (0, 1e-4]")
    if ratio.numel() == 0 or not torch.isfinite(ratio).all():
        return False, float("inf")
    maximum_deviation = float((ratio - 1).abs().max().detach().cpu())
    passed = bool(
        torch.allclose(
            ratio, torch.ones_like(ratio), rtol=0.0, atol=absolute_tolerance
        )
    )
    return passed, maximum_deviation


__all__ = [
    "clipped_grpo_surrogate",
    "flowmol_grpo_step",
    "group_relative_advantages",
    "initial_ratio_diagnostic",
]
