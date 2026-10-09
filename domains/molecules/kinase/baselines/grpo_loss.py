"""FlowGRPO primitives for FlowMol3's mixed continuous/discrete sampler.

The coordinate transition follows the ODE-to-SDE construction from Flow-GRPO
(arXiv:2505.05470), expressed in FlowMol's noise-to-data time convention.  A
FlowMol coordinate endpoint prediction ``x_1`` implies

    v_t = (x_1 - x_t) / (1 - t)
    score_t = -(x_t - t x_1) / (1 - t)^2

for its released linear interpolant.  The same-marginal SDE drift is
``v_t + 0.5 * sigma_t**2 * score_t``.  FlowMol's atom, charge, and bond heads
are categorical CTMC endpoint policies, so their sampled endpoint tokens are
included as discrete actions in the joint transition policy.

As in the authors' implementation, log densities are averaged over feature
dimensions before forming the PPO ratio.  This avoids an unusably sharp ratio
for high-dimensional generative actions and makes the clip range an explicit
algorithmic convention rather than a molecule-size-dependent accident.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor


def group_relative_advantages(
    rewards: Tensor,
    group_ids: Sequence[object],
    *,
    epsilon: float = 1e-4,
    clip_max: float | None = 5.0,
) -> Tensor:
    """Return population-standardized rewards within each GRPO group."""

    values = rewards.detach().reshape(-1)
    if values.numel() != len(group_ids) or not values.numel():
        raise ValueError("each non-empty reward needs exactly one group identifier")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("FlowGRPO rewards must be finite")
    advantages = torch.empty_like(values)
    for group in dict.fromkeys(group_ids):
        positions = [index for index, value in enumerate(group_ids) if value == group]
        index = torch.tensor(positions, dtype=torch.long, device=values.device)
        members = values[index]
        advantages[index] = (members - members.mean()) / (
            members.std(unbiased=False) + epsilon
        )
    if clip_max is not None:
        if clip_max <= 0:
            raise ValueError("advantage clip must be positive")
        advantages = advantages.clamp(-clip_max, clip_max)
    return advantages


def flowgrpo_noise_std(
    time: Tensor | float,
    *,
    step_size: Tensor | float,
    noise_level: float,
) -> Tensor:
    """Flow-GRPO diffusion coefficient in noise-to-data time.

    The paper uses ``a * sqrt(t_noise / (1 - t_noise))`` while integrating
    from data to noise.  FlowMol integrates in the opposite convention, so
    ``t_noise = 1 - time``.  At the exact prior endpoint the denominator is
    regularized by one integration step, matching the official implementation's
    use of the first interior noise level for its singular endpoint.
    """

    if noise_level <= 0 or not math.isfinite(noise_level):
        raise ValueError("noise level must be finite and positive")
    time_tensor = torch.as_tensor(time)
    step_tensor = torch.as_tensor(step_size, device=time_tensor.device, dtype=time_tensor.dtype)
    if bool((step_tensor <= 0).any()):
        raise ValueError("step size must be positive")
    effective_time = torch.maximum(time_tensor, step_tensor).clamp(max=1.0)
    ratio = (1.0 - effective_time).clamp_min(0.0) / effective_time.clamp_min(
        torch.finfo(effective_time.dtype).eps
    )
    return noise_level * torch.sqrt(ratio)


def coordinate_transition_mean(
    x_t: Tensor,
    endpoint: Tensor,
    *,
    time_per_node: Tensor,
    step_size: float,
    noise_std_per_node: Tensor,
) -> Tensor:
    """Euler-Maruyama mean for the FlowMol coordinate SDE transition."""

    if x_t.shape != endpoint.shape or x_t.ndim != 2:
        raise ValueError("coordinate state and endpoint must be equal 2-D tensors")
    if time_per_node.ndim != 1 or time_per_node.numel() != x_t.shape[0]:
        raise ValueError("one scalar time is required per node")
    if noise_std_per_node.ndim != 1 or noise_std_per_node.numel() != x_t.shape[0]:
        raise ValueError("one scalar noise coefficient is required per node")
    remaining = (1.0 - time_per_node).clamp_min(1e-6).unsqueeze(-1)
    time = time_per_node.unsqueeze(-1)
    sigma = noise_std_per_node.unsqueeze(-1)
    velocity = (endpoint - x_t) / remaining
    score = -(x_t - time * endpoint) / remaining.square()
    drift = velocity + 0.5 * sigma.square() * score
    return x_t + float(step_size) * drift


def _mean_per_graph(values: Tensor, batch_index: Tensor, graph_count: int) -> Tensor:
    if values.ndim != 1 or batch_index.ndim != 1:
        raise ValueError("segment values and indices must be one-dimensional")
    if values.numel() != batch_index.numel():
        raise ValueError("segment values and indices must have equal length")
    totals = torch.zeros(graph_count, device=values.device, dtype=values.dtype)
    counts = torch.zeros(graph_count, device=values.device, dtype=values.dtype)
    totals.scatter_add_(0, batch_index, values)
    counts.scatter_add_(0, batch_index, torch.ones_like(values))
    if bool((counts == 0).any()):
        raise ValueError("every graph must contribute at least one action element")
    return totals / counts


def gaussian_log_prob_per_graph(
    action: Tensor,
    mean: Tensor,
    *,
    standard_deviation: Tensor,
    batch_index: Tensor,
    graph_count: int,
) -> Tensor:
    """Average isotropic-Gaussian log density per coordinate and graph."""

    if action.shape != mean.shape or action.ndim != 2:
        raise ValueError("Gaussian action and mean must be equal 2-D tensors")
    if standard_deviation.ndim != 1 or standard_deviation.numel() != action.shape[0]:
        raise ValueError("one Gaussian standard deviation is required per node")
    std = standard_deviation.clamp_min(torch.finfo(action.dtype).eps).unsqueeze(-1)
    elementwise = -0.5 * ((action.detach() - mean) / std).square()
    elementwise = elementwise - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    return _mean_per_graph(elementwise.mean(dim=-1), batch_index, graph_count)


def categorical_log_prob_per_graph(
    probabilities: Tensor,
    actions: Tensor,
    *,
    batch_index: Tensor,
    graph_count: int,
) -> Tensor:
    """Average categorical action log probability for each molecule."""

    if probabilities.ndim != 2 or actions.ndim != 1:
        raise ValueError("categorical probabilities/actions have invalid ranks")
    if probabilities.shape[0] != actions.numel():
        raise ValueError("categorical action count differs from probability rows")
    selected = probabilities.clamp_min(1e-12).gather(1, actions[:, None]).squeeze(1)
    return _mean_per_graph(selected.log(), batch_index, graph_count)


def categorical_kl_per_graph(
    policy_probabilities: Tensor,
    reference_probabilities: Tensor,
    *,
    batch_index: Tensor,
    graph_count: int,
) -> Tensor:
    """Mean KL(policy || reference) for one categorical modality."""

    if policy_probabilities.shape != reference_probabilities.shape:
        raise ValueError("policy/reference categorical distributions must match")
    policy = policy_probabilities.clamp_min(1e-12)
    reference = reference_probabilities.clamp_min(1e-12)
    per_element = (policy * (policy.log() - reference.log())).sum(dim=-1)
    return _mean_per_graph(per_element, batch_index, graph_count)


def joint_transition_statistic(components: Sequence[Tensor]) -> Tensor:
    """Average modality-level transition statistics into one policy scalar."""

    if not components:
        raise ValueError("at least one transition component is required")
    shapes = {tuple(component.shape) for component in components}
    if len(shapes) != 1 or components[0].ndim != 1:
        raise ValueError("transition components must be equal per-graph vectors")
    return torch.stack(tuple(components), dim=0).mean(dim=0)


def clipped_grpo_loss(
    new_log_prob: Tensor,
    old_log_prob: Tensor,
    advantage: Tensor,
    *,
    clip_range: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """PPO-clipped GRPO policy loss and detached diagnostics."""

    shapes = {tuple(value.shape) for value in (new_log_prob, old_log_prob, advantage)}
    if len(shapes) != 1 or new_log_prob.ndim != 1 or not new_log_prob.numel():
        raise ValueError("FlowGRPO expects equal non-empty per-graph vectors")
    if clip_range <= 0:
        raise ValueError("clip range must be positive")
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    unclipped = -advantage * ratio
    clipped = -advantage * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = torch.maximum(unclipped, clipped).mean()
    return loss, {
        "ratio": ratio.detach(),
        "approx_kl": (0.5 * log_ratio.square()).mean().detach(),
        "clip_fraction": (ratio.sub(1.0).abs() > clip_range).float().mean().detach(),
    }


__all__ = [
    "categorical_kl_per_graph",
    "categorical_log_prob_per_graph",
    "clipped_grpo_loss",
    "coordinate_transition_mean",
    "flowgrpo_noise_std",
    "gaussian_log_prob_per_graph",
    "group_relative_advantages",
    "joint_transition_statistic",
]
