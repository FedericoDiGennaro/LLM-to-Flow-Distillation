"""Reward normalization and implicit-policy utilities for FlowMol NFT.

The molecular adapter is nft_update.flowmol_nft_step. It forms implicit
predictions on FlowMol's coordinate and categorical outputs and uses the
registered multi-modal flow-matching loss. This is a molecular adaptation,
not the original image benchmark or an exact likelihood objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

Normalizer = Literal["global_std", "group_std"]


def optimality_reward(
    raw_rewards: Tensor,
    group_ids: Sequence[Any],
    *,
    normalizer: Normalizer = "global_std",
    epsilon: float = 1e-8,
) -> Tensor:
    """Map raw rewards to optimality probabilities in [0, 1] (Algorithm 1).

    Rewards are centred within their group, scaled and clipped. Both ordering
    and relative reward gaps matter. A group whose members all score alike yields 0.5
    everywhere, which contributes no improvement direction -- the correct
    outcome when the reward cannot separate the candidates.
    """

    values = raw_rewards.detach().reshape(-1).double()
    if values.numel() != len(group_ids):
        raise ValueError("each reward needs exactly one group identifier")
    if not values.numel():
        raise ValueError("optimality reward requires at least one sample")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("raw rewards must be finite")
    centred = torch.empty_like(values)
    scales = torch.empty_like(values)
    for group in dict.fromkeys(group_ids):
        index = torch.tensor(
            [position for position, key in enumerate(group_ids) if key == group],
            dtype=torch.long,
        )
        members = values[index]
        centred[index] = members - members.mean()
        if normalizer == "group_std":
            scales[index] = members.std(unbiased=False)
    if normalizer == "global_std":
        scales = torch.full_like(values, float(centred.std(unbiased=False)))
    elif normalizer != "group_std":
        raise ValueError("normalizer must be 'global_std' or 'group_std'")
    scaled = centred / scales.clamp_min(epsilon)
    return (0.5 + 0.5 * scaled.clamp(-1.0, 1.0)).to(raw_rewards.dtype)


def implicit_policies(
    policy_prediction: Tensor, reference_prediction: Tensor, *, beta: float
) -> tuple[Tensor, Tensor]:
    """Return the implicit positive and negative predictions (Theorem 3.2).

    The positive policy interpolates from the sampling policy toward the
    trained one and the negative extrapolates away from it, which is what turns
    a single trained network into a contrast between two implicit policies.
    """

    if beta <= 0:
        raise ValueError("beta must be positive")
    if policy_prediction.shape != reference_prediction.shape:
        raise ValueError("policy and reference predictions must have equal shape")
    positive = (1 - beta) * reference_prediction + beta * policy_prediction
    negative = (1 + beta) * reference_prediction - beta * policy_prediction
    return positive, negative


def negative_aware_loss(
    positive_loss: Tensor, negative_loss: Tensor, optimality: Tensor
) -> Tensor:
    """Combine the two branches of Eq. (5) using the optimality probability.

    Dropping the negative branch recovers rejection fine-tuning, which the
    paper reports collapses almost immediately for diffusion.
    """

    shapes = {
        tuple(value.shape) for value in (positive_loss, negative_loss, optimality)
    }
    if len(shapes) != 1:
        raise ValueError("per-sample losses and optimality must have the same shape")
    if positive_loss.ndim != 1 or not positive_loss.numel():
        raise ValueError("DiffusionNFT requires a non-empty vector of samples")
    if bool((optimality < 0).any()) or bool((optimality > 1).any()):
        raise ValueError("optimality probabilities must lie in [0, 1]")
    return (optimality * positive_loss + (1 - optimality) * negative_loss).mean()


def soft_update_(collection_policy: Any, trained_policy: Any, *, eta: float) -> None:
    """In-place EMA of the data-collection policy (Algorithm 1, line 14).

    eta = 0 is fully on-policy and eta -> 1 is nearly offline; the paper warms
    up from a small value because a strictly on-policy schedule collapses.
    """

    if not 0.0 <= eta < 1.0:
        raise ValueError("eta must lie in [0, 1)")
    trained_state = trained_policy.state_dict()
    with torch.no_grad():
        for name, value in collection_policy.state_dict().items():
            update = trained_state[name]
            # Entries the update never touched are left byte-identical. Blending
            # equal values is a no-op mathematically but not in floating point,
            # and drifting them would break the frozen-state audit.
            if torch.equal(value, update):
                continue
            if not value.is_floating_point():
                value.copy_(update)
                continue
            value.mul_(eta).add_(update.to(value.dtype), alpha=1 - eta)


def soft_update_schedule(iteration: int, *, slope: float, maximum: float) -> float:
    """eta_i = min(slope * i, maximum), the schedule used in the paper."""

    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if not 0.0 <= maximum < 1.0 or slope < 0:
        raise ValueError("schedule must stay within [0, 1)")
    return min(slope * iteration, maximum)


@dataclass(frozen=True)
class NFTPreflightResult:
    ready: bool
    blockers: tuple[str, ...]


def validate_nft_preflight(
    contract: Mapping[str, Any], manifest: Mapping[str, Any]
) -> NFTPreflightResult:
    """Fail closed unless the rollout binds to a reward and a trainable state."""

    blockers: list[str] = []
    if contract.get("algorithm") != "diffusion_nft_v1":
        blockers.append("unsupported_algorithm")
    if contract.get("reward_model_id") != manifest.get("reward_model_id"):
        blockers.append("reward_model_mismatch")
    if contract.get("collection_checkpoint") != manifest.get("collection_checkpoint"):
        blockers.append("collection_policy_mismatch")
    if not manifest.get("fresh_rollout"):
        blockers.append("not_fresh_rollout")
    if not manifest.get("training_state_bindings_verified"):
        blockers.append("training_state_bindings_unverified")
    # A reward model that also scores the final report would make the baseline
    # trivially win its own evaluation.
    held_out = manifest.get("held_out_evaluator_ids", [])
    if not isinstance(held_out, list):
        blockers.append("malformed_held_out_evaluator_ids")
        held_out = []
    held_out = [*held_out, manifest.get("held_out_evaluator_id")]
    if manifest.get("reward_model_id") in held_out:
        blockers.append("reward_model_is_the_held_out_evaluator")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        blockers.append("sample_list_missing")
        samples = []
    if len(samples) < int(contract.get("minimum_sample_count", 1)):
        blockers.append("too_few_samples")
    groups: set[Any] = set()
    scored = 0
    for sample in samples:
        if not isinstance(sample, Mapping):
            blockers.append("malformed_sample")
            continue
        if not str(sample.get("content_sha256", "")).strip():
            blockers.append("unbound_sample_content")
        group = sample.get("group_id")
        if not isinstance(group, str) or not group.strip():
            blockers.append("group_identifiers_missing")
        else:
            groups.add(group)
        reward = sample.get("raw_reward")
        if (not isinstance(reward, (int, float)) or isinstance(reward, bool)
                or not math.isfinite(reward)):
            blockers.append("sample_without_numeric_reward")
            continue
        scored += 1
    if samples and scored == len(samples) and len(groups) < 1:
        blockers.append("group_identifiers_missing")
    minimum_groups = int(contract.get("minimum_group_count", 1))
    if len(groups) < minimum_groups:
        blockers.append("too_few_reward_groups")
    # Failed assemblies must be scored at the floor, not silently dropped, or
    # the baseline is rewarded for producing fewer but prettier candidates.
    if manifest.get("failed_candidates_scored") is not True:
        blockers.append("failed_candidates_not_scored")
    return NFTPreflightResult(False if blockers else True, tuple(sorted(set(blockers))))


__all__ = [
    "NFTPreflightResult",
    "implicit_policies",
    "negative_aware_loss",
    "optimality_reward",
    "soft_update_",
    "soft_update_schedule",
    "validate_nft_preflight",
]
