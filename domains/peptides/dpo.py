"""All-pairs preference optimization for the pinned PepDFM flow model.

This is a registered flow-DPO surrogate, not an exact sequence-likelihood DPO
implementation.  For every stable-positive/stable-negative Cartesian pair it
compares the policy and frozen-parent differences in PepDFM's per-sequence
``MixturePathGeneralizedKL`` loss under identical corruptions.  Lower flow
loss is treated as higher implicit sequence preference.

For preferred sequence ``w`` and rejected sequence ``l`` the minimized loss is

    -log sigmoid(-0.5 * beta * ((L_pi(w) - L_pi(l))
                                - (L_ref(w) - L_ref(l)))).

The parent checkpoint at the start of each generator update is the frozen
reference.  Abstentions and unstable labels never enter the pair set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from domains.peptides.model import PEPDFM_VOCABULARY_SIZE
from domains.peptides.checkpoint import (
    PepDfmCheckpoint,
    sha256_state_dict,
)


FLOW_DPO_KIND = "all_cartesian_pairs_pepdfm_flow_dpo"


@dataclass(frozen=True)
class FlowDPOConfig:
    """Bounded controls for one all-pairs flow-DPO update."""

    learning_rate: float = 1e-5
    steps: int = 100
    batch_size: int = 32
    beta: float = 2.6
    gradient_clip_norm: float = 0.1
    epsilon: float = 1e-3
    seed: int = 8427
    reference_policy: str = "parent_checkpoint_per_update"
    pair_construction: str = "full_cartesian_stable_positive_x_stable_negative"
    pair_schedule: str = "deterministic_shuffled_epochs_without_replacement"
    corruption_coupling: str = "same_policy_reference_corruption_per_sequence"
    loss_kind: str = "diffusion_dpo_logistic_on_generalized_kl_surrogate"

    def __post_init__(self) -> None:
        for name in ("learning_rate", "beta", "gradient_clip_norm", "epsilon"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.epsilon >= 1:
            raise ValueError("epsilon must be smaller than one")
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
                "same_policy_reference_corruption_per_sequence"
            ),
            "loss_kind": (
                "diffusion_dpo_logistic_on_generalized_kl_surrogate"
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} must remain {value!r}")


@dataclass(frozen=True)
class PreferencePair:
    """One immutable preferred/rejected sequence pairing."""

    pair_id: str
    preferred_sequence: str
    rejected_sequence: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "preferred_sequence": self.preferred_sequence,
            "preferred_length": len(self.preferred_sequence),
            "rejected_sequence": self.rejected_sequence,
            "rejected_length": len(self.rejected_sequence),
        }


def cartesian_preference_pairs(
    positive_sequences: Sequence[str], negative_sequences: Sequence[str]
) -> tuple[PreferencePair, ...]:
    """Enumerate the complete deterministic P x N preference relation."""

    positives = tuple(sorted(set(positive_sequences)))
    negatives = tuple(sorted(set(negative_sequences)))
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
                    preferred_sequence=preferred,
                    rejected_sequence=rejected,
                )
            )
    return tuple(pairs)


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
    positive_sequences: Sequence[str],
    negative_sequences: Sequence[str],
) -> dict[str, Any]:
    """Publish the explicit full Cartesian pair set, or verify exact re-entry."""

    pairs = cartesian_preference_pairs(positive_sequences, negative_sequences)
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
        "positive_count": len(set(positive_sequences)),
        "negative_count": len(set(negative_sequences)),
        "construction": "full_cartesian_stable_positive_x_stable_negative",
    }


def load_and_validate_pair_manifest(
    path: Path,
    *,
    positive_sequences: Sequence[str],
    negative_sequences: Sequence[str],
) -> tuple[PreferencePair, ...]:
    expected = cartesian_preference_pairs(positive_sequences, negative_sequences)
    expected_text = canonical_pair_jsonl(expected)
    if not path.is_file() or path.read_text(encoding="utf-8") != expected_text:
        raise ValueError("flow-DPO pair manifest is not the exact Cartesian set")
    return expected


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


def _per_example_loss(
    model: PepDfmCheckpoint,
    *,
    sequences: Sequence[str],
    pair_times: torch.Tensor,
    reference: PepDfmCheckpoint,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return policy/reference losses with identical per-sequence corruption."""

    if len(sequences) != int(pair_times.shape[0]):
        raise ValueError("sequence/timestep batch size differs")
    policy_values = torch.empty(
        len(sequences), dtype=torch.float32, device=model.device
    )
    reference_values = torch.empty_like(policy_values)
    by_length: dict[int, list[int]] = {}
    for index, sequence in enumerate(sequences):
        by_length.setdefault(len(sequence), []).append(index)
    loss_fn = model._loss_class(path=model.path, reduction="none")
    for _length, indices in sorted(by_length.items()):
        group_sequences = tuple(sequences[index] for index in indices)
        x_1 = model._tokens(group_sequences)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=model.device)
        times = pair_times[index_tensor]
        x_0 = torch.randint(
            low=0,
            high=PEPDFM_VOCABULARY_SIZE,
            size=x_1.shape,
            dtype=x_1.dtype,
            device=x_1.device,
        )
        path_sample = model.path.sample(t=times, x_0=x_0, x_1=x_1)
        policy_logits = model.model(path_sample.x_t, path_sample.t)
        policy_loss = loss_fn(
            logits=policy_logits,
            x_1=x_1,
            x_t=path_sample.x_t,
            t=path_sample.t,
        )
        with torch.no_grad():
            reference_logits = reference.model(path_sample.x_t, path_sample.t)
            reference_loss = loss_fn(
                logits=reference_logits,
                x_1=x_1,
                x_t=path_sample.x_t,
                t=path_sample.t,
            )
        while policy_loss.ndim > 1:
            policy_loss = policy_loss.mean(dim=-1)
            reference_loss = reference_loss.mean(dim=-1)
        policy_values[index_tensor] = policy_loss
        reference_values[index_tensor] = reference_loss
    return policy_values, reference_values


def fine_tune_flow_dpo(
    *,
    policy: PepDfmCheckpoint,
    reference: PepDfmCheckpoint,
    positive_sequences: Sequence[str],
    negative_sequences: Sequence[str],
    pair_manifest_path: Path,
    config: FlowDPOConfig,
) -> dict[str, Any]:
    """Optimize every Cartesian stable-positive/stable-negative pair."""

    pairs = load_and_validate_pair_manifest(
        pair_manifest_path,
        positive_sequences=positive_sequences,
        negative_sequences=negative_sequences,
    )
    schedule = deterministic_pair_schedule(
        pair_count=len(pairs),
        steps=config.steps,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    policy_before = sha256_state_dict(policy.model)
    reference_before = sha256_state_dict(reference.model)
    if policy_before != reference_before:
        raise ValueError("flow-DPO policy must initialize from its reference parent")
    for parameter in reference.model.parameters():
        parameter.requires_grad_(False)
    reference.model.eval()
    policy.model.train()
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=config.learning_rate)
    random_devices = [policy.device] if policy.device.type == "cuda" else []
    visited = [0] * len(pairs)
    losses: list[float] = []
    accuracies: list[float] = []
    margins: list[float] = []
    winner_deltas: list[float] = []
    rejected_deltas: list[float] = []
    maximum_grad = 0.0
    with torch.random.fork_rng(devices=random_devices):
        torch.manual_seed(config.seed)
        for step in range(config.steps):
            start = step * config.batch_size
            indices = schedule[start : start + config.batch_size]
            batch = tuple(pairs[index] for index in indices)
            for index in indices:
                visited[index] += 1
            pair_times = torch.rand(
                len(batch), device=policy.device
            ) * (1.0 - config.epsilon)
            preferred_policy, preferred_reference = _per_example_loss(
                policy,
                sequences=tuple(row.preferred_sequence for row in batch),
                pair_times=pair_times,
                reference=reference,
            )
            rejected_policy, rejected_reference = _per_example_loss(
                policy,
                sequences=tuple(row.rejected_sequence for row in batch),
                pair_times=pair_times,
                reference=reference,
            )
            policy_difference = preferred_policy - rejected_policy
            reference_difference = preferred_reference - rejected_reference
            implicit_margin = -(policy_difference - reference_difference)
            logits = 0.5 * config.beta * implicit_margin
            loss = -F.logsigmoid(logits).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite flow-DPO loss at step {step}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.model.parameters(), config.gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite flow-DPO gradient norm at step {step}"
                )
            maximum_grad = max(maximum_grad, float(gradient_norm))
            optimizer.step()
            losses.append(float(loss.detach()))
            accuracies.append(float((implicit_margin.detach() > 0).float().mean()))
            margins.append(float(implicit_margin.detach().mean()))
            winner_deltas.append(
                float((preferred_policy - preferred_reference).detach().mean())
            )
            rejected_deltas.append(
                float((rejected_policy - rejected_reference).detach().mean())
            )
    policy.model.eval()
    policy_after = sha256_state_dict(policy.model)
    reference_after = sha256_state_dict(reference.model)
    if policy_after == policy_before:
        raise RuntimeError("flow-DPO fine-tuning did not change the PepDFM state")
    if reference_after != reference_before:
        raise RuntimeError("flow-DPO reference checkpoint changed during training")
    if min(visited) < 1:
        raise RuntimeError("flow-DPO failed to visit every Cartesian pair")
    return {
        "objective": FLOW_DPO_KIND,
        "positive_count": len(set(positive_sequences)),
        "negative_count": len(set(negative_sequences)),
        "pair_count": len(pairs),
        "pair_manifest_path": str(pair_manifest_path.resolve()),
        "pair_manifest_sha256": pair_manifest_sha256(pairs),
        "pair_slots_consumed": len(schedule),
        "all_pairs_visited": True,
        "minimum_pair_visit_count": min(visited),
        "maximum_pair_visit_count": max(visited),
        "complete_pair_epochs": len(schedule) // len(pairs),
        "config": asdict(config),
        # These are stochastic training-batch diagnostics.  In particular, the
        # first and last values use different pairs/corruptions and are not a
        # fixed-set before/after evaluation.
        "first_batch_dpo_loss": losses[0],
        "last_batch_dpo_loss": losses[-1],
        "minimum_training_batch_dpo_loss": min(losses),
        "maximum_training_batch_dpo_loss": max(losses),
        "first_batch_preference_accuracy": accuracies[0],
        "last_batch_preference_accuracy": accuracies[-1],
        "last_batch_mean_implicit_margin": margins[-1],
        "last_batch_mean_preferred_loss_change_vs_reference": winner_deltas[-1],
        "last_batch_mean_rejected_loss_change_vs_reference": rejected_deltas[-1],
        "training_batch_diagnostics_are_not_fixed_set_pre_post_metrics": True,
        "maximum_preclip_gradient_norm": maximum_grad,
        "checkpoint_sha256_before": policy_before,
        "checkpoint_sha256_after": policy_after,
        "reference_checkpoint_sha256_before": reference_before,
        "reference_checkpoint_sha256_after": reference_after,
        "reference_checkpoint_unchanged": True,
        "flow_dpo_surrogate_not_exact_sequence_log_likelihood": True,
    }
