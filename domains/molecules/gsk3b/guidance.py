"""Differentiable endpoint surrogate used for GSK3B-only FlowMol3 guidance."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


def _batch_mean(values: torch.Tensor, batch: torch.Tensor, size: int) -> torch.Tensor:
    result = torch.zeros((size, values.shape[-1]), device=values.device, dtype=values.dtype)
    result.index_add_(0, batch, values)
    counts = torch.bincount(batch, minlength=size).to(values.dtype).clamp_min(1)
    return result / counts.unsqueeze(-1)


def endpoint_features(
    *, x: torch.Tensor, a: torch.Tensor, c: torch.Tensor, e: torch.Tensor,
    node_batch: torch.Tensor, edge_batch: torch.Tensor, batch_size: int,
) -> torch.Tensor:
    """Pooled soft graph features available at every endpoint prediction."""

    atom = _batch_mean(a, node_batch, batch_size)
    charge = _batch_mean(c, node_batch, batch_size)
    bond = _batch_mean(e, edge_batch, batch_size)
    center = _batch_mean(x, node_batch, batch_size)[node_batch]
    radial = torch.linalg.vector_norm(x - center, dim=-1, keepdim=True)
    radius_mean = _batch_mean(radial, node_batch, batch_size)
    radius_sq_mean = _batch_mean(radial.square(), node_batch, batch_size)
    radius_sd = (radius_sq_mean - radius_mean.square()).clamp_min(0).sqrt()
    counts = torch.bincount(node_batch, minlength=batch_size).to(x.dtype)
    count_feature = torch.log1p(counts).unsqueeze(-1) / 5.0
    return torch.cat([atom, charge, bond, radius_mean, radius_sd, count_feature], dim=-1)


def molecule_features(molecules: list[Any]) -> torch.Tensor:
    rows = []
    for molecule in molecules:
        graph = molecule.g
        upper = graph.edata["ue_mask"] if "ue_mask" in graph.edata else None
        if upper is None:
            from flowmol.data_processing.utils import get_upper_edge_mask
            upper = get_upper_edge_mask(graph)
        x = graph.ndata["x_1"].float()
        a = graph.ndata["a_1"].float()
        c = graph.ndata["c_1"].float()
        e = graph.edata["e_1"][upper].float()
        # FlowMol's CTMC integrator appends a transient mask category to each
        # terminal categorical tensor. The vector field's differentiable
        # endpoint predictions contain only the modeled categories. Distill on
        # that same feature space so fitting and guidance are dimensionally
        # identical; judgeable molecules cannot retain a selected mask token.
        if bool(getattr(molecule, "ctmc_mol", False)):
            if min(a.shape[-1], c.shape[-1], e.shape[-1]) < 2:
                raise ValueError("CTMC endpoint lacks removable mask categories")
            a = a[..., :-1]
            c = c[..., :-1]
            e = e[..., :-1]
        rows.append(endpoint_features(
            x=x, a=a, c=c, e=e,
            node_batch=torch.zeros(x.shape[0], dtype=torch.long),
            edge_batch=torch.zeros(e.shape[0], dtype=torch.long),
            batch_size=1,
        )[0])
    return torch.stack(rows)


class EndpointRewardSurrogate(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features).squeeze(-1))


def load_surrogate(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = EndpointRewardSurrogate(
        int(checkpoint["feature_dim"]), int(checkpoint["hidden_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    mean = checkpoint["feature_mean"].to(device)
    scale = checkpoint["feature_scale"].to(device)
    return model, mean, scale, checkpoint


@contextmanager
def guided_endpoint_predictions(
    vector_field: Any,
    surrogate: EndpointRewardSurrogate,
    mean: torch.Tensor,
    scale: torch.Tensor,
    *,
    categorical_scale: float,
    coordinate_scale: float,
) -> Iterator[None]:
    """Temporarily guide predicted endpoints by the distilled reward gradient."""

    original = vector_field.forward

    def guided(*args: Any, **kwargs: Any):
        raw = original(*args, **kwargs)
        graph = args[0] if args else kwargs["g"]
        node_batch = kwargs.get("node_batch_idx", args[2] if len(args) > 2 else None)
        upper = kwargs.get("upper_edge_mask", args[3] if len(args) > 3 else None)
        if node_batch is None or upper is None:
            raise RuntimeError("guided FlowMol forward lacks batch bindings")
        from flowmol.data_processing.utils import get_edge_batch_idxs
        edge_batch = get_edge_batch_idxs(graph)[upper]
        with torch.enable_grad():
            leaves = {
                key: raw[key].detach().requires_grad_(True)
                for key in ("x", "a", "c", "e")
            }
            features = endpoint_features(
                x=leaves["x"], a=leaves["a"], c=leaves["c"], e=leaves["e"],
                node_batch=node_batch, edge_batch=edge_batch,
                batch_size=int(graph.batch_size),
            )
            reward = surrogate((features - mean) / scale).sum()
            gradients = torch.autograd.grad(reward, tuple(leaves.values()))
        result = dict(raw)
        result["x"] = (leaves["x"] + coordinate_scale * gradients[0]).detach()
        for index, key in enumerate(("a", "c", "e"), start=1):
            logits = torch.log(leaves[key].clamp_min(1e-8))
            result[key] = torch.softmax(
                logits + categorical_scale * gradients[index], dim=-1
            ).detach()
        return result

    vector_field.forward = guided
    try:
        yield
    finally:
        vector_field.forward = original


__all__ = [
    "EndpointRewardSurrogate",
    "endpoint_features",
    "guided_endpoint_predictions",
    "load_surrogate",
    "molecule_features",
]
