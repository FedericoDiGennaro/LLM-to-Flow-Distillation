"""All-pairs flow-DPO on retained FlowMol3 native graphs.

This is a registered flow-matching DPO surrogate, not an exact molecular
likelihood DPO.  For every stable-positive/stable-negative Cartesian pair it
compares the policy and frozen-parent differences in FlowMol3's per-molecule
multi-modal flow-matching loss under identical interpolants.  Lower flow
loss is treated as higher implicit preference.

For preferred molecule ``w`` and rejected molecule ``l`` the minimized loss is

    -log sigmoid(-0.5 * beta * ((L_pi(w) - L_pi(l))
                                - (L_ref(w) - L_ref(l)))).

The parent checkpoint at the start of each generator update is the frozen
reference.  Abstentions and unstable labels never enter the pair set.
Distort-on-path noise from FlowMol3 training is not applied; interpolants are
coupled exactly between policy and reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


FLOW_DPO_KIND = "all_cartesian_pairs_flowmol_flow_dpo"


@dataclass(frozen=True)
class FlowDPOConfig:
    """Bounded controls for one all-pairs FlowMol flow-DPO update."""

    learning_rate: float = 1e-5
    steps: int = 100
    batch_size: int = 8
    beta: float = 1.0
    gradient_clip_norm: float = 1.0
    seed: int = 30657
    reference_policy: str = "parent_checkpoint_per_update"
    pair_construction: str = "full_cartesian_stable_positive_x_stable_negative"
    pair_schedule: str = "deterministic_shuffled_epochs_without_replacement"
    corruption_coupling: str = "same_policy_reference_interpolant_per_molecule"
    loss_kind: str = "diffusion_dpo_logistic_on_flowmol_flow_matching_surrogate"

    def __post_init__(self) -> None:
        for name in ("learning_rate", "beta", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("steps", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        expected = {
            "reference_policy": "parent_checkpoint_per_update",
            "pair_construction": (
                "full_cartesian_stable_positive_x_stable_negative"
            ),
            "pair_schedule": (
                "deterministic_shuffled_epochs_without_replacement"
            ),
            "corruption_coupling": (
                "same_policy_reference_interpolant_per_molecule"
            ),
            "loss_kind": (
                "diffusion_dpo_logistic_on_flowmol_flow_matching_surrogate"
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} must remain {value!r}")


@dataclass(frozen=True)
class PreferencePair:
    """One immutable preferred/rejected molecule pairing."""

    pair_id: str
    preferred_item_id: str
    rejected_item_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "preferred_item_id": self.preferred_item_id,
            "rejected_item_id": self.rejected_item_id,
        }


def cartesian_preference_pairs(
    positive_item_ids: Sequence[str], negative_item_ids: Sequence[str]
) -> tuple[PreferencePair, ...]:
    """Enumerate the complete deterministic P x N preference relation."""

    positives = tuple(sorted(set(positive_item_ids)))
    negatives = tuple(sorted(set(negative_item_ids)))
    if not positives or not negatives:
        raise ValueError("flow-DPO requires non-empty positive and negative sets")
    if set(positives).intersection(negatives):
        raise ValueError("positive and negative preference sets must be disjoint")
    pairs: list[PreferencePair] = []
    for positive_index, preferred in enumerate(positives):
        for negative_index, rejected in enumerate(negatives):
            pairs.append(
                PreferencePair(
                    pair_id=(
                        f"pair-p{positive_index:04d}-n{negative_index:04d}"
                    ),
                    preferred_item_id=preferred,
                    rejected_item_id=rejected,
                )
            )
    return tuple(pairs)


def cap_preference_pairs(
    pairs: Sequence[PreferencePair], *, maximum: int, seed: int
) -> tuple[PreferencePair, ...]:
    """Keep a deterministic subset when the Cartesian product exceeds the cap."""

    if maximum <= 0:
        raise ValueError("pair cap must be positive")
    if len(pairs) <= maximum:
        return tuple(pairs)
    ranked = sorted(
        pairs,
        key=lambda pair: hashlib.sha256(
            f"{seed}:{pair.preferred_item_id}:{pair.rejected_item_id}".encode()
        ).hexdigest(),
    )
    return tuple(ranked[:maximum])


def preference_pairs_for_training(
    positive_item_ids: Sequence[str],
    negative_item_ids: Sequence[str],
    *,
    maximum: int | None,
    seed: int,
) -> tuple[PreferencePair, ...]:
    pairs = cartesian_preference_pairs(positive_item_ids, negative_item_ids)
    if maximum is None:
        return pairs
    return cap_preference_pairs(pairs, maximum=maximum, seed=seed)


def canonical_pair_jsonl(pairs: Sequence[PreferencePair]) -> str:
    return "".join(
        json.dumps(pair.as_dict(), allow_nan=False, sort_keys=True) + "\n"
        for pair in pairs
    )


def pair_manifest_sha256(pairs: Sequence[PreferencePair]) -> str:
    return hashlib.sha256(canonical_pair_jsonl(pairs).encode("utf-8")).hexdigest()


def write_or_validate_pair_manifest(
    path: Path,
    *,
    positive_item_ids: Sequence[str],
    negative_item_ids: Sequence[str],
    maximum: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Publish the training pair set, or verify exact re-entry."""

    pairs = cartesian_preference_pairs(positive_item_ids, negative_item_ids)
    cartesian_count = len(pairs)
    if maximum is not None:
        pairs = cap_preference_pairs(pairs, maximum=maximum, seed=seed)
    text = canonical_pair_jsonl(pairs)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError("immutable flow-DPO pair manifest differs")
    else:
        temporary = path.with_name(f".{path.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        temporary.write_text(text, encoding="utf-8")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError("immutable flow-DPO pair manifest differs")
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "pair_count": len(pairs),
        "cartesian_pair_count": cartesian_count,
        "positive_count": len(set(positive_item_ids)),
        "negative_count": len(set(negative_item_ids)),
        "construction": (
            "full_cartesian_stable_positive_x_stable_negative_capped"
            if len(pairs) < cartesian_count
            else "full_cartesian_stable_positive_x_stable_negative"
        ),
    }


def load_and_validate_pair_manifest(
    path: Path,
    *,
    positive_item_ids: Sequence[str],
    negative_item_ids: Sequence[str],
) -> tuple[PreferencePair, ...]:
    expected = cartesian_preference_pairs(positive_item_ids, negative_item_ids)
    expected_text = canonical_pair_jsonl(expected)
    if not path.is_file() or path.read_text(encoding="utf-8") != expected_text:
        raise ValueError("flow-DPO pair manifest is not the exact Cartesian set")
    return expected


def optimizer_microbatch_count(optimizer_steps: int, accumulation: int) -> int:
    """Map registered Adam steps onto the inner flow-DPO microbatch loop."""

    if isinstance(optimizer_steps, bool) or isinstance(accumulation, bool):
        raise ValueError("optimizer steps and accumulation must be integers")
    if optimizer_steps <= 0 or accumulation <= 0:
        raise ValueError("optimizer steps and accumulation must be positive")
    return optimizer_steps * accumulation


def deterministic_pair_schedule(
    *, pair_count: int, steps: int, batch_size: int, seed: int
) -> tuple[int, ...]:
    """Traverse one full shuffled epoch, then deterministic partial epochs."""

    if pair_count <= 0 or steps <= 0 or batch_size <= 0:
        raise ValueError("pair schedule dimensions must be positive")
    slots = steps * batch_size
    if slots < pair_count:
        raise ValueError(
            "registered flow-DPO schedule cannot visit every Cartesian pair"
        )
    schedule: list[int] = []
    epoch = 0
    while len(schedule) < slots:
        order = list(range(pair_count))
        random.Random(seed + epoch).shuffle(order)
        schedule.extend(order)
        epoch += 1
    return tuple(schedule[:slots])


def flow_dpo_loss(
    policy_chosen_loss: Tensor,
    policy_rejected_loss: Tensor,
    reference_chosen_loss: Tensor,
    reference_rejected_loss: Tensor,
    *,
    beta: float,
) -> Tensor:
    """Apply the peptide flow-DPO logistic to per-molecule flow-matching losses."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    values = (
        policy_chosen_loss,
        policy_rejected_loss,
        reference_chosen_loss,
        reference_rejected_loss,
    )
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("all per-molecule loss tensors must have the same shape")
    if policy_chosen_loss.ndim != 1 or not policy_chosen_loss.numel():
        raise ValueError("flow-DPO requires a non-empty pair vector")
    policy_difference = policy_chosen_loss - policy_rejected_loss
    reference_difference = reference_chosen_loss - reference_rejected_loss
    implicit_margin = -(policy_difference - reference_difference)
    return -F.logsigmoid(0.5 * beta * implicit_margin).mean()


def drop_mask_one_hot(tensor: Tensor, n_classes: int) -> Tensor:
    """Strip the CTMC mask token so destination one-hots match the training loader."""

    if n_classes <= 0:
        raise ValueError("n_classes must be positive")
    width = int(tensor.shape[-1])
    if width == n_classes:
        return tensor.float()
    if width == n_classes + 1:
        return tensor[..., :n_classes].float()
    raise ValueError(
        f"categorical feature width {width} is neither {n_classes} nor {n_classes + 1}"
    )


def copy_dgl_graph(graph: Any) -> Any:
    import dgl

    src, dst = graph.edges(form="uv")
    copied = dgl.graph((src, dst), num_nodes=graph.num_nodes(), device=graph.device)
    for name in graph.ndata.keys():
        copied.ndata[name] = graph.ndata[name].detach().clone()
    for name in graph.edata.keys():
        copied.edata[name] = graph.edata[name].detach().clone()
    if int(graph.batch_size) > 1:
        copied.set_batch_num_nodes(
            graph.batch_num_nodes().detach().clone().to(copied.device)
        )
        copied.set_batch_num_edges(
            graph.batch_num_edges().detach().clone().to(copied.device)
        )
    if int(copied.batch_size) != int(graph.batch_size):
        raise RuntimeError("copied FlowMol graph lost per-molecule batch membership")
    return copied


def training_graph_from_sampled(molecule: Any, model: Any) -> Any:
    """Rebuild a FlowMol training graph from a retained native SampledMolecule."""

    graph = getattr(molecule, "g", None)
    if graph is None:
        raise ValueError("sampled molecule is missing its native FlowMol graph")
    copied = copy_dgl_graph(graph)
    if "x_1" not in copied.ndata:
        raise ValueError("sampled graph is missing destination coordinates")
    positions = copied.ndata["x_1"].float()
    copied.ndata["x_1_true"] = positions - positions.mean(dim=0, keepdim=True)
    copied.ndata["a_1_true"] = drop_mask_one_hot(
        copied.ndata["a_1"], int(model.n_cat_dict["a"])
    )
    copied.ndata["c_1_true"] = drop_mask_one_hot(
        copied.ndata["c_1"], int(model.n_cat_dict["c"])
    )
    copied.edata["e_1_true"] = drop_mask_one_hot(
        copied.edata["e_1"], int(model.n_cat_dict["e"])
    )
    return copied


def attach_prior(model: Any, graph: Any) -> Any:
    """Sample FlowMol priors onto a training graph in place."""

    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    node_batch_idx, _edge_batch_idx = get_batch_idxs(graph)
    upper_edge_mask = get_upper_edge_mask(graph)
    return model.sample_prior(graph, node_batch_idx, upper_edge_mask)


def interpolate_conditional_path(model: Any, graph: Any, times: Tensor) -> Any:
    """Build one interpolant for a batched graph at the given per-molecule times."""

    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    if times.ndim != 1 or int(times.shape[0]) != int(graph.batch_size):
        raise ValueError("interpolant times must be one scalar per molecule")
    copied = copy_dgl_graph(graph)
    if int(copied.batch_size) != int(times.shape[0]):
        raise RuntimeError("interpolant copy does not keep one graph per molecule")
    node_batch_idx, edge_batch_idx = get_batch_idxs(copied)
    upper_edge_mask = get_upper_edge_mask(copied)
    return model.vector_field.sample_conditional_path(
        copied, times, node_batch_idx, edge_batch_idx, upper_edge_mask
    )


def _scatter_mean(values: Tensor, index: Tensor, batch_size: int) -> Tensor:
    flat = values.reshape(values.shape[0], -1).mean(dim=-1)
    sums = torch.zeros(batch_size, device=flat.device, dtype=flat.dtype)
    counts = torch.zeros(batch_size, device=flat.device, dtype=flat.dtype)
    sums.index_add_(0, index, flat)
    counts.index_add_(0, index, torch.ones_like(flat))
    return sums / counts.clamp_min(1.0)


def _feature_loss_fn(model: Any, feat: str) -> Any:
    from torch import nn

    if feat == "x":
        return nn.MSELoss(reduction="none")
    kwargs: dict[str, Any] = {"reduction": "none"}
    if model.parameterization == "ctmc":
        kwargs["ignore_index"] = -100
    return nn.CrossEntropyLoss(**kwargs)


def _apply_time_weight(unreduced: Tensor, weight: Tensor) -> Tensor:
    if unreduced.ndim == 1:
        return unreduced * weight
    shape = [-1] + [1] * (unreduced.ndim - 1)
    return unreduced * weight.reshape(*shape)


def per_graph_flow_loss(
    model: Any,
    interpolated: Any,
    times: Tensor,
    vf_output: Mapping[str, Tensor] | None = None,
) -> Tensor:
    """Score an already-interpolated batch; does not resample the interpolant.

    Passing ``vf_output`` substitutes another policy's predictions into the
    same weighted loss, which is how DiffusionNFT scores its implicit positive
    and negative policies without a second interpolant.
    """

    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    batch_size = int(interpolated.batch_size)
    node_batch_idx, edge_batch_idx = get_batch_idxs(interpolated)
    upper_edge_mask = get_upper_edge_mask(interpolated)
    if vf_output is None:
        vf_output = model.vector_field(
            interpolated,
            times,
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
        )
    time_weights = None
    if model.time_scaled_loss:
        time_weights = model.interpolant_scheduler.loss_weights(times)

    per_graph = torch.zeros(batch_size, device=interpolated.device)
    for feat_idx, feat in enumerate(model.canonical_feat_order):
        data_src = interpolated.edata if feat == "e" else interpolated.ndata
        target = data_src[f"{feat}_1_true"]
        if feat == "e":
            target = target[upper_edge_mask]
        if feat in ["a", "c", "e"]:
            target = target.argmax(dim=-1)
            if model.parameterization == "ctmc":
                xt = data_src[f"{feat}_t"]
                xt_idxs = (
                    xt[upper_edge_mask].argmax(-1) if feat == "e" else xt.argmax(-1)
                )
                target = target.clone()
                target[xt_idxs != model.n_cat_dict[feat]] = -100
        unreduced = _feature_loss_fn(model, feat)(vf_output[feat], target)
        if time_weights is not None:
            weight = time_weights[:, feat_idx]
            if feat == "e":
                weight = weight[edge_batch_idx][upper_edge_mask]
            else:
                weight = weight[node_batch_idx]
            unreduced = _apply_time_weight(unreduced, weight)
        if feat == "e":
            graph_loss = _scatter_mean(
                unreduced, edge_batch_idx[upper_edge_mask], batch_size
            )
        else:
            graph_loss = _scatter_mean(unreduced, node_batch_idx, batch_size)
        per_graph = per_graph + float(model.total_loss_weights[feat]) * graph_loss
    return per_graph


def _batch_graphs(graphs: Sequence[Any]) -> Any:
    import dgl

    if not graphs:
        raise ValueError("cannot batch an empty molecule set")
    return dgl.batch([copy_dgl_graph(graph) for graph in graphs])


def per_example_loss(
    policy: Any,
    reference: Any,
    *,
    molecules: Sequence[Any],
    pair_times: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return policy/reference losses with one shared interpolant per molecule."""

    if len(molecules) != int(pair_times.shape[0]):
        raise ValueError("molecule/timestep batch size differs")
    prepared = []
    for molecule in molecules:
        graph = training_graph_from_sampled(molecule, policy)
        graph = attach_prior(policy, graph)
        prepared.append(graph)
    batched = _batch_graphs(prepared).to(pair_times.device)
    interpolated = interpolate_conditional_path(policy, batched, pair_times)
    policy_loss = per_graph_flow_loss(policy, copy_dgl_graph(interpolated), pair_times)
    with torch.no_grad():
        reference_loss = per_graph_flow_loss(
            reference, copy_dgl_graph(interpolated), pair_times
        )
    return policy_loss, reference_loss


def sha256_state_dict(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def relative_parameter_delta(
    policy: Any, reference: Any, trainable_names: Sequence[str]
) -> float:
    policy_parameters = dict(policy.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    baseline = sum(
        float(reference_parameters[name].detach().double().pow(2).sum())
        for name in trainable_names
    )
    delta = sum(
        float(
            (
                policy_parameters[name].detach().double()
                - reference_parameters[name].detach().double()
            )
            .pow(2)
            .sum()
        )
        for name in trainable_names
    )
    return math.sqrt(delta) / max(math.sqrt(baseline), 1e-12)


def verify_frozen_state(
    policy: Any, reference: Any, trainable: Sequence[Tensor]
) -> list[str]:
    selected_ids = {id(parameter) for parameter in trainable}
    mutable_names = {
        name
        for name, parameter in policy.named_parameters(remove_duplicate=False)
        if id(parameter) in selected_ids
    }
    reference_state = reference.state_dict()
    for name, value in policy.state_dict().items():
        if name not in mutable_names and not torch.equal(value, reference_state[name]):
            raise RuntimeError(f"frozen FlowMol state changed: {name}")
    return sorted(mutable_names)


def save_reloadable_checkpoint(
    *,
    parent_checkpoint: Path,
    policy: Any,
    destination: Path,
) -> None:
    """Write a Lightning checkpoint the released FlowMol loader can reload."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        str(parent_checkpoint), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, Mapping) or "state_dict" not in checkpoint:
        raise ValueError("parent FlowMol file is not a Lightning checkpoint")
    checkpoint["state_dict"] = {
        name: value.detach().cpu().contiguous()
        for name, value in policy.state_dict().items()
    }
    torch.save(checkpoint, destination)


def config_from_campaign(training: Mapping[str, Any]) -> FlowDPOConfig:
    return FlowDPOConfig(
        learning_rate=float(training["learning_rate"]),
        steps=int(training["steps"]),
        batch_size=int(training["batch_size_pairs"]),
        beta=float(training["beta"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        seed=int(training["seed"]),
    )


__all__ = [
    "_batch_graphs",
    "FLOW_DPO_KIND",
    "FlowDPOConfig",
    "PreferencePair",
    "attach_prior",
    "canonical_pair_jsonl",
    "cap_preference_pairs",
    "cartesian_preference_pairs",
    "config_from_campaign",
    "copy_dgl_graph",
    "deterministic_pair_schedule",
    "drop_mask_one_hot",
    "flow_dpo_loss",
    "interpolate_conditional_path",
    "load_and_validate_pair_manifest",
    "optimizer_microbatch_count",
    "per_example_loss",
    "per_graph_flow_loss",
    "preference_pairs_for_training",
    "relative_parameter_delta",
    "save_reloadable_checkpoint",
    "sha256_state_dict",
    "training_graph_from_sampled",
    "verify_frozen_state",
    "write_or_validate_pair_manifest",
]
