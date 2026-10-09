"""Differentiable soft-graph reward model for FlowMol reward guidance.

Registered terminal rewards need not provide gradients through FlowMol's mixed
continuous/discrete state.  This module implements the small time-conditioned
regressor used to distil such an oracle onto FlowMol's soft endpoint
predictions.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn


def time_features(times: torch.Tensor) -> torch.Tensor:
    """Return a bounded four-channel embedding for times in [0, 1]."""

    values = times.float().reshape(-1, 1)
    return torch.cat(
        (
            values,
            values.square(),
            torch.sin(torch.pi * values),
            torch.cos(torch.pi * values),
        ),
        dim=-1,
    )


def graph_rms(
    values: torch.Tensor,
    batch_index: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    """RMS of all non-batch entries, independently for every graph."""

    if values.ndim < 2:
        raise ValueError("graph_rms expects at least two dimensions")
    squared = values.square().reshape(values.shape[0], -1).mean(dim=-1)
    totals = torch.zeros(graph_count, device=values.device, dtype=values.dtype)
    counts = torch.zeros_like(totals)
    totals.index_add_(0, batch_index, squared)
    counts.index_add_(0, batch_index, torch.ones_like(squared))
    return (totals / counts.clamp_min(1.0)).clamp_min(0.0).sqrt()


def normalized_graph_gradient(
    gradient: torch.Tensor,
    batch_index: torch.Tensor,
    graph_count: int,
    *,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Normalize one gradient tensor to unit RMS within each molecule."""

    rms = graph_rms(gradient, batch_index, graph_count)
    return gradient / rms.clamp_min(epsilon)[batch_index].reshape(
        (-1,) + (1,) * (gradient.ndim - 1)
    )


class SoftGraphRewardRegressor(nn.Module):
    """Invariant message-passing regressor over soft molecular graphs.

    Coordinates are deliberately excluded.  Both registered uses—the TDC
    DRD2/LogP oracle and the strict kinase topology/geometry endpoint—guide only
    the categorical graph heads; FlowMol's native coordinate dynamics remain
    untouched.
    """

    def __init__(
        self,
        *,
        atom_dim: int,
        charge_dim: int,
        bond_dim: int,
        hidden_dim: int = 96,
        message_layers: int = 3,
    ) -> None:
        super().__init__()
        if min(atom_dim, charge_dim, bond_dim, hidden_dim, message_layers) < 1:
            raise ValueError("reward-regressor dimensions must be positive")
        self.atom_dim = int(atom_dim)
        self.charge_dim = int(charge_dim)
        self.bond_dim = int(bond_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_layers = int(message_layers)
        time_dim = 4
        self.node_input = nn.Sequential(
            nn.Linear(atom_dim + charge_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.message_mlps = nn.ModuleList()
        self.update_mlps = nn.ModuleList()
        for _ in range(message_layers):
            self.message_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim + bond_dim + time_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            )
            self.update_mlps.append(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(message_layers)
        )
        self.output = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def architecture(self) -> dict[str, int]:
        return {
            "atom_dim": self.atom_dim,
            "charge_dim": self.charge_dim,
            "bond_dim": self.bond_dim,
            "hidden_dim": self.hidden_dim,
            "message_layers": self.message_layers,
        }

    def forward(
        self,
        *,
        atom_probabilities: torch.Tensor,
        charge_probabilities: torch.Tensor,
        bond_probabilities: torch.Tensor,
        node_batch_index: torch.Tensor,
        upper_edge_source: torch.Tensor,
        upper_edge_destination: torch.Tensor,
        upper_edge_batch_index: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        graph_count = int(times.numel())
        if graph_count < 1:
            raise ValueError("reward regressor requires at least one graph")
        if atom_probabilities.shape[0] != node_batch_index.shape[0]:
            raise ValueError("atom probabilities and node batch index differ")
        if charge_probabilities.shape[0] != node_batch_index.shape[0]:
            raise ValueError("charge probabilities and node batch index differ")
        if bond_probabilities.shape[0] != upper_edge_source.shape[0]:
            raise ValueError("bond probabilities and upper edges differ")

        time = time_features(times)
        node_time = time[node_batch_index]
        hidden = self.node_input(
            torch.cat((atom_probabilities, charge_probabilities, node_time), dim=-1)
        )
        # FlowMol stores one representative from each symmetric edge pair in
        # its endpoint distribution.  Pass messages in both directions.
        source = torch.cat((upper_edge_source, upper_edge_destination), dim=0)
        destination = torch.cat((upper_edge_destination, upper_edge_source), dim=0)
        edge_features = torch.cat((bond_probabilities, bond_probabilities), dim=0)
        # Edge class zero is "no bond" in FlowMol.  Without this weight, the
        # O(N^2) non-bond pairs overwhelm the O(N) chemical bonds and the
        # regressor mostly learns molecule size.  Soft bond mass keeps the
        # operation differentiable at noisy intermediate states.
        bonded_mass = (1.0 - edge_features[:, 0]).clamp(0.0, 1.0)
        edge_time = torch.cat(
            (time[upper_edge_batch_index], time[upper_edge_batch_index]), dim=0
        )
        for message_mlp, update_mlp, norm in zip(
            self.message_mlps, self.update_mlps, self.norms
        ):
            messages = message_mlp(
                torch.cat(
                    (
                        hidden[source],
                        hidden[destination],
                        edge_features,
                        edge_time,
                    ),
                    dim=-1,
                )
            )
            messages = messages * bonded_mass.unsqueeze(-1)
            aggregate = torch.zeros_like(hidden)
            aggregate.index_add_(0, destination, messages)
            degree = torch.zeros(
                hidden.shape[0], device=hidden.device, dtype=hidden.dtype
            )
            degree.index_add_(0, destination, bonded_mass)
            aggregate = aggregate / degree.clamp_min(1.0).unsqueeze(-1)
            hidden = norm(hidden + update_mlp(torch.cat((hidden, aggregate), dim=-1)))

        pooled_sum = torch.zeros(
            graph_count, self.hidden_dim, device=hidden.device, dtype=hidden.dtype
        )
        pooled_square = torch.zeros_like(pooled_sum)
        counts = torch.zeros(graph_count, device=hidden.device, dtype=hidden.dtype)
        pooled_sum.index_add_(0, node_batch_index, hidden)
        pooled_square.index_add_(0, node_batch_index, hidden.square())
        counts.index_add_(0, node_batch_index, torch.ones_like(node_batch_index, dtype=hidden.dtype))
        mean = pooled_sum / counts.clamp_min(1.0).unsqueeze(-1)
        variance = (
            pooled_square / counts.clamp_min(1.0).unsqueeze(-1) - mean.square()
        ).clamp_min(1e-8)
        size = torch.log1p(counts).unsqueeze(-1)
        return self.output(torch.cat((mean, variance.sqrt(), time, size), dim=-1)).squeeze(-1)


def guided_categorical_probabilities(
    *,
    predictor: SoftGraphRewardRegressor,
    endpoint: Mapping[str, torch.Tensor],
    node_batch_index: torch.Tensor,
    upper_edge_source: torch.Tensor,
    upper_edge_destination: torch.Tensor,
    upper_edge_batch_index: torch.Tensor,
    times: torch.Tensor,
    temperature: torch.Tensor | float,
    strengths: Mapping[str, float],
    schedule_power: float,
    maximum_logit_shift: float,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Tilt FlowMol endpoint-token policies along the reward gradient."""

    graph_count = int(times.numel())
    logits = {
        feature: endpoint[feature].detach().clamp_min(1e-12).log().requires_grad_(True)
        for feature in ("a", "c", "e")
    }
    probabilities = {feature: value.softmax(dim=-1) for feature, value in logits.items()}
    prediction = predictor(
        atom_probabilities=probabilities["a"],
        charge_probabilities=probabilities["c"],
        bond_probabilities=probabilities["e"],
        node_batch_index=node_batch_index,
        upper_edge_source=upper_edge_source,
        upper_edge_destination=upper_edge_destination,
        upper_edge_batch_index=upper_edge_batch_index,
        times=times,
    )
    gradients = torch.autograd.grad(prediction.sum(), tuple(logits.values()))
    schedule = times.clamp(0.0, 1.0).pow(float(schedule_power))
    batch_indices = {
        "a": node_batch_index,
        "c": node_batch_index,
        "e": upper_edge_batch_index,
    }
    guided: dict[str, torch.Tensor] = {}
    shift_rms: dict[str, float] = {}
    scalar_temperature = torch.as_tensor(
        temperature, device=times.device, dtype=endpoint["a"].dtype
    ).clamp_min(1e-6)
    for (feature, base_logits), gradient in zip(logits.items(), gradients):
        batch_index = batch_indices[feature]
        normalized = normalized_graph_gradient(gradient, batch_index, graph_count)
        multiplier = (
            schedule[batch_index]
            * float(strengths[feature])
        ).reshape((-1,) + (1,) * (gradient.ndim - 1))
        shift = (normalized * multiplier).clamp(
            min=-float(maximum_logit_shift), max=float(maximum_logit_shift)
        )
        # FlowMol temperature is applied to the base distribution first.  The
        # bounded guidance shift then changes sampling odds on an interpretable
        # logit scale instead of being amplified by the very low temperature.
        guided[feature] = (base_logits.detach() / scalar_temperature + shift).softmax(dim=-1)
        shift_rms[feature] = float(shift.detach().square().mean().sqrt().cpu())
    diagnostics = {
        "prediction_mean": float(prediction.detach().mean().cpu()),
        "atom_logit_shift_rms": shift_rms["a"],
        "charge_logit_shift_rms": shift_rms["c"],
        "bond_logit_shift_rms": shift_rms["e"],
    }
    return guided, diagnostics
