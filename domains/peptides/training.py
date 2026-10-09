"""Run one chained PepDFM alignment/resampling transition.

The entry point deliberately executes exactly one pending round.  A round is
pending when ``rounds/round-NN/alignment_buffer.json`` exists but its immutable
``training/summary.json`` does not.  Round 00 starts from the released PepDFM
checkpoint; every later round is cryptographically chained to the preceding
round's gated child checkpoint.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import torch
from domains.peptides.replay import contiguous_same_goal_prior_indices
from domains.peptides.model import PepDfmAssets
from domains.peptides.dpo import FLOW_DPO_KIND, FlowDPOConfig, fine_tune_flow_dpo
from domains.peptides.checkpoint import (
    PepDfmCheckpoint,
    encode_peptide,
    sha256_file,
    sha256_state_dict,
    stratified_hash_split,
)
from domains.peptides.metrics import jensen_shannon, sample_rows, summarize_samples
from domains.peptides.predictor_evidence import (
    attach_joint_within_length_developability_evidence,
)
from adm.campaign import resolve_campaign_registration

ROUND_NAME = re.compile("round-(\\d{2})")


@dataclass(frozen=True)
class PendingRound:
    index: int
    root: Path
    buffer_path: Path
    candidate_path: Path
    training_root: Path


@dataclass(frozen=True)
class ParentCheckpoint:
    kind: str
    path: Path | None
    expected_sha256: str | None
    source_round_index: int | None
    source_summary_path: Path | None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or not all((isinstance(row, dict) for row in rows)):
        raise ValueError(f"expected non-empty JSON-object rows: {path}")
    return rows


def immutable_write(path: Path, text: str) -> None:
    """Create a text artifact once, accepting only byte-identical re-entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"immutable artifact differs: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def persist_validated_inputs(
    training_root: Path,
    *,
    validated: Mapping[str, Any],
    execution_authorization: Mapping[str, Any],
) -> None:
    """Persist scientific inputs independently from per-attempt authorization.

    Scientific inputs are immutable across retries. Authorization provenance is
    intentionally attempt-specific: a diagnosed repair changes the approved
    commit and active-campaign hash without changing labels, seeds, or training
    data. Older runs embedded authorization inside ``validated_inputs.json``;
    accept that legacy layout only when its remaining scientific payload is
    identical, and preserve the old file byte-for-byte.
    """
    scientific = dict(validated)
    scientific.pop("execution_authorization", None)
    scientific_text = (
        json.dumps(scientific, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    scientific_normalized = json.loads(scientific_text)
    validated_path = training_root / "validated_inputs.json"
    if validated_path.exists():
        existing = read_json(validated_path)
        existing.pop("execution_authorization", None)
        if existing != scientific_normalized:
            raise FileExistsError(
                f"immutable scientific inputs differ: {validated_path}"
            )
    else:
        immutable_write(validated_path, scientific_text)
    commit = str(execution_authorization.get("code_commit", ""))
    campaign_sha = str(execution_authorization.get("active_campaign_sha256", ""))
    if not re.fullmatch("[0-9a-f]{40}", commit) or not re.fullmatch(
        "[0-9a-f]{64}", campaign_sha
    ):
        raise ValueError("execution authorization lacks canonical provenance hashes")
    attempt_path = (
        training_root
        / "authorization_attempts"
        / f"{commit[:12]}-{campaign_sha[:12]}.json"
    )
    immutable_write(
        attempt_path,
        json.dumps(
            {
                "schema_version": 1,
                "validated_inputs_path": str(validated_path.resolve()),
                "execution_authorization": dict(execution_authorization),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def validate_training_contract(config: Mapping[str, Any]) -> None:
    """Reject drift away from a registered replay or flow-DPO update."""
    tuning = config.get("fine_tuning")
    if not isinstance(tuning, Mapping):
        raise TypeError("fine_tuning must be an object")
    kind = tuning.get("kind")
    if kind != FLOW_DPO_KIND:
        raise ValueError("fine_tuning.kind must be all_cartesian_pairs_pepdfm_flow_dpo")
    alpha = float(tuning.get("negative_alpha", math.nan))
    if alpha != 0.0:
        raise ValueError("flow-DPO requires negative_alpha to remain zero")
    if float(tuning.get("holdout_fraction", math.nan)) != 0.0:
        raise ValueError("flow-DPO must pair every stable positive and negative")
    expected = {
        "beta": 2.6,
        "reference_policy": "parent_checkpoint_per_update",
        "pair_construction": "full_cartesian_stable_positive_x_stable_negative",
        "pair_schedule": "deterministic_shuffled_epochs_without_replacement",
        "corruption_coupling": "same_policy_reference_corruption_per_sequence",
        "loss_kind": "diffusion_dpo_logistic_on_generalized_kl_surrogate",
    }
    for name, expected_value in expected.items():
        observed = tuning.get(name)
        if isinstance(expected_value, float):
            matches = math.isclose(
                float(observed), expected_value, rel_tol=0.0, abs_tol=0.0
            )
        else:
            matches = observed == expected_value
        if not matches:
            raise ValueError(
                f"flow-DPO fine_tuning.{name} must remain {expected_value!r}"
            )
    gates = config.get("checkpoint_gates")
    if not isinstance(gates, Mapping):
        raise TypeError("checkpoint_gates must be an object")
    require_loss_gates = kind == "positive_only_pepdfm_generalized_kl"
    for field in (
        "require_positive_holdout_loss_decrease",
        "require_negative_minus_positive_margin_non_decrease",
    ):
        if gates.get(field) is not require_loss_gates:
            raise ValueError(
                f"checkpoint_gates.{field} must be {require_loss_gates} for {kind}"
            )


def _git_state(repository_root: Path) -> dict[str, Any]:

    def command(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=repository_root, text=True
        ).strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain")),
    }


def _validate_runtime_config_binding(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    resources: Mapping[str, Any],
    repository_root: Path,
) -> str:
    observed_config_hash = sha256_file(config_path)
    template_name = resources.get("config_template")
    if template_name is None:
        if resources.get("config_sha256") != observed_config_hash:
            raise PermissionError("approved config SHA-256 differs from the job config")
        registered = (repository_root / str(resources.get("config", ""))).resolve()
        if registered != config_path.resolve():
            raise PermissionError("job config path differs from the approved resource")
        return observed_config_hash
    template_path = (repository_root / str(template_name)).resolve()
    if sha256_file(template_path) != str(resources.get("config_template_sha256", "")):
        raise PermissionError("approved V12 template SHA-256 differs")
    if (
        config_path.resolve()
        != Path(str(resources.get("derived_config_path", ""))).resolve()
    ):
        raise PermissionError("V12 derived config path differs from registration")
    template = read_json(template_path)
    template_pool = template.get("candidate_pool")
    config_pool = config.get("candidate_pool")
    if not isinstance(template_pool, dict) or not isinstance(config_pool, Mapping):
        raise PermissionError("V12 candidate-pool binding is malformed")
    if template_pool.get("sha256") != "0" * 64:
        raise PermissionError("V12 template has no zero candidate binding")
    candidate_path = Path(
        os.path.expandvars(str(config_pool.get("path", "")))
    ).resolve()
    candidate_hash = sha256_file(candidate_path)
    expected = dict(template)
    expected["candidate_pool"] = dict(template_pool)
    expected["candidate_pool"]["sha256"] = candidate_hash
    protocol_version = template.get("prompt_protocol", {}).get("version")
    geometric_v13 = protocol_version == "geometric-semantic-reachability-v13"
    moving_frontier = protocol_version in {
        "moving-semantic-frontier-v12b",
        "moving-semantic-frontier-v12c",
        "moving-semantic-frontier-v12d",
        "geometric-semantic-reachability-v13",
    }
    reference_hash: str | None = None
    policy_hash: str | None = None
    length_counts: dict[str, int] | None = None
    if geometric_v13:
        template_policy = template.get("campaign_length_policy")
        config_policy = config.get("campaign_length_policy")
        if not isinstance(template_policy, Mapping) or not isinstance(
            config_policy, Mapping
        ):
            raise PermissionError("V13 campaign length-policy binding is malformed")
        if template_policy.get("sha256") != "0" * 64 or template_policy.get(
            "length_counts"
        ) not in ({}, None):
            raise PermissionError("V13 template length-policy placeholders differ")
        policy_path = Path(
            os.path.expandvars(str(config_policy.get("path", "")))
        ).resolve()
        policy_hash = sha256_file(policy_path)
        policy_artifact = read_json(policy_path)
        allocations = policy_artifact.get("length_policy", {}).get("allocations")
        if not isinstance(allocations, list) or not allocations:
            raise PermissionError("V13 policy artifact omitted allocations")
        length_counts = {
            str(int(row["length"])): int(row["count"])
            for row in allocations
            if isinstance(row, Mapping)
        }
        if (
            len(length_counts) != len(allocations)
            or sum(length_counts.values()) != 100
            or policy_artifact.get("policy_frozen_for_entire_campaign") is not True
            or (config_policy.get("sha256") != policy_hash)
            or (config_policy.get("length_counts") != length_counts)
            or (config_policy.get("selected_before_p0") is not True)
            or (config_policy.get("frozen_for_entire_campaign") is not True)
        ):
            raise PermissionError("V13 frozen campaign length policy differs")
        expected["campaign_length_policy"] = {
            **dict(template_policy),
            "sha256": policy_hash,
            "length_counts": length_counts,
            "selected_before_p0": True,
            "frozen_for_entire_campaign": True,
        }
        expected["candidate_pool"]["length_counts"] = length_counts
        if (
            template.get("anticancer_reference_context") is not None
            or config.get("anticancer_reference_context") is not None
            or template.get("prompt_protocol", {}).get(
                "labelled_reference_context_visible"
            )
            is not False
            or (
                template.get("prompt_protocol", {}).get(
                    "external_peptide_dataset_visible"
                )
                is not False
            )
        ):
            raise PermissionError(
                "V13 must not bind or expose an anticancer reference dataset"
            )
    if moving_frontier:
        template_sampling = template.get("fresh_sampling")
        config_sampling = config.get("fresh_sampling")
        if not isinstance(template_sampling, Mapping) or not isinstance(
            config_sampling, Mapping
        ):
            raise PermissionError("moving-frontier sampling binding is malformed")
        template_reference = template_sampling.get("frozen_evidence_reference")
        config_reference = config_sampling.get("frozen_evidence_reference")
        if not isinstance(template_reference, Mapping) or not isinstance(
            config_reference, Mapping
        ):
            raise PermissionError("moving-frontier reference binding is malformed")
        if template_reference.get("sha256") != "0" * 64:
            raise PermissionError(
                "moving-frontier template has no zero evidence-reference binding"
            )
        reference_path = Path(
            os.path.expandvars(str(config_reference.get("path", "")))
        ).resolve()
        reference_hash = sha256_file(reference_path)
        expected["fresh_sampling"] = dict(template_sampling)
        if geometric_v13:
            if template_sampling.get("length_counts_per_arm") not in ({}, None):
                raise PermissionError("V13 template length counts must be empty")
            assert length_counts is not None
            expected["fresh_sampling"]["length_counts_per_arm"] = length_counts
        expected["fresh_sampling"]["frozen_evidence_reference"] = dict(
            template_reference
        )
        expected["fresh_sampling"]["frozen_evidence_reference"]["sha256"] = (
            reference_hash
        )
    if dict(config) != expected or config_pool.get("sha256") != candidate_hash:
        raise PermissionError("runtime config changed beyond approved hash bindings")
    manifest = read_json(candidate_path.parent / "pool_manifest.json")
    if manifest.get("candidate_pool_sha256") != candidate_hash:
        raise PermissionError("V12 pool manifest candidate binding differs")
    if manifest.get("config_sha256") != resources.get("pool_config_sha256"):
        raise PermissionError("V12 pool manifest config binding differs")
    if geometric_v13 and (
        manifest.get("length_policy_sha256") != policy_hash
        or manifest.get("external_dataset_context_visible") is not False
        or manifest.get("length_histogram") != length_counts
    ):
        raise PermissionError("V13 pool manifest dynamic-artifact binding differs")
    frozen_manifest = read_json(config_path.with_name("frozen_config_manifest.json"))
    if frozen_manifest.get("frozen_config_sha256") != observed_config_hash:
        raise PermissionError("V12 frozen config manifest differs")
    if (
        moving_frontier
        and frozen_manifest.get("frozen_evidence_reference_sha256") != reference_hash
    ):
        raise PermissionError("moving-frontier frozen evidence binding differs")
    if geometric_v13 and (
        frozen_manifest.get("campaign_length_policy_sha256") != policy_hash
        or frozen_manifest.get("external_dataset_context_visible") is not False
        or frozen_manifest.get("length_counts") != length_counts
    ):
        raise PermissionError("V13 frozen config manifest differs")
    return observed_config_hash


def validate_execution_authorization(
    *, config_path: Path, output_root: Path, campaign_path: Path, repository_root: Path
) -> dict[str, Any]:
    """Bind a GPU round to the human-approved campaign, commit, and config.

    The campaign lives outside this experiment worktree, so the job validates
    it before creating any round artifact.  The exact config bytes, clean Git
    commit/branch, output root, deadline, and three-job GPU envelope must all
    match the approved record.
    """
    campaign = read_json(campaign_path)
    if campaign.get("schema_version") != 1:
        raise PermissionError("active campaign schema differs from registration")
    config = read_json(config_path)
    experiment_id = str(config.get("experiment_id", ""))
    if campaign.get("status") != "active":
        raise PermissionError("ACTIVE_CAMPAIGN.json does not authorize this experiment")
    registration = resolve_campaign_registration(campaign, experiment_id=experiment_id)
    approval_gate = registration.approval_gate
    if (
        not isinstance(approval_gate, Mapping)
        or approval_gate.get("status") != "approved"
    ):
        raise PermissionError("GPU approval gate is not approved")
    resources = registration.resources
    state = _git_state(repository_root)
    if resources.get("code_root") != repository_root.name:
        raise PermissionError("approved code root differs from the GPU worktree")
    allowed_code_roots = campaign.get("allowed_code_roots")
    if (
        not isinstance(allowed_code_roots, list)
        or repository_root.name not in allowed_code_roots
    ):
        raise PermissionError("GPU worktree is outside the approved code roots")
    observed_config_hash = _validate_runtime_config_binding(
        config=config,
        config_path=config_path,
        resources=resources,
        repository_root=repository_root,
    )
    if str(resources.get("output_root", "")) != str(output_root.resolve()):
        raise PermissionError("GPU output root differs from the approved resource")
    if resources.get("code_commit") != state["commit"]:
        raise PermissionError("GPU worktree commit differs from the approved commit")
    if resources.get("code_branch") != state["branch"]:
        raise PermissionError("GPU worktree branch differs from the approved branch")
    if state["dirty"]:
        raise PermissionError("GPU execution requires a clean approved worktree")
    budget = campaign.get("budget")
    if not isinstance(budget, Mapping):
        raise PermissionError("active campaign has no compute budget")
    deadline = dt.datetime.fromisoformat(str(budget["deadline"]).replace("Z", "+00:00"))
    if dt.datetime.now(dt.timezone.utc) > deadline:
        raise PermissionError("campaign deadline has expired")
    registered_budget = config.get("registered_compute_budget")
    if registered_budget is None:
        registered_budget = {
            "max_jobs": 4,
            "max_gpu_hours": 2.0,
            "max_parallel_jobs": 1,
        }
    if not isinstance(registered_budget, Mapping):
        raise PermissionError("registered_compute_budget must be an object")
    required_budget = {
        "max_jobs": int(registered_budget["max_jobs"]),
        "max_gpu_hours": float(registered_budget["max_gpu_hours"]),
        "max_parallel_jobs": int(registered_budget["max_parallel_jobs"]),
    }
    for name, required in required_budget.items():
        try:
            available = float(budget[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise PermissionError(f"active campaign has invalid {name}") from exc
        if available < float(required):
            raise PermissionError(
                f"active campaign {name} does not cover the registered requirement"
            )
    return {
        "active_campaign_path": str(campaign_path.resolve()),
        "active_campaign_sha256": sha256_file(campaign_path),
        "config_sha256": observed_config_hash,
        "code_commit": state["commit"],
        "code_branch": state["branch"],
        "output_root": str(output_root.resolve()),
        "deadline": str(budget["deadline"]),
        "checkpoint_recovery_binding": resources.get("checkpoint_recovery_binding"),
    }


def discover_pending_round(output_root: Path, *, maximum_rounds: int) -> PendingRound:
    """Return the sole pending round and reject ambiguous or gapped ledgers."""
    rounds_root = output_root.resolve() / "rounds"
    with_buffers: dict[int, Path] = {}
    if rounds_root.exists():
        for path in rounds_root.iterdir():
            match = ROUND_NAME.fullmatch(path.name)
            if not path.is_dir() or match is None:
                continue
            index = int(match.group(1))
            buffer_path = path / "alignment_buffer.json"
            if buffer_path.is_file():
                if index >= maximum_rounds:
                    raise ValueError(
                        f"round {index:02d} exceeds the registered alignment-round ceiling"
                    )
                with_buffers[index] = path
    pending = [
        (index, root)
        for index, root in sorted(with_buffers.items())
        if not (root / "training" / "summary.json").exists()
    ]
    if len(pending) != 1:
        raise ValueError(
            f"expected exactly one pending alignment round, found {[index for index, _root in pending]}"
        )
    index, root = pending[0]
    if sorted(with_buffers) != list(range(index + 1)):
        raise ValueError(
            "round ledger has a gap or a future buffer before the pending round"
        )
    for prior_index in range(index):
        prior_summary = with_buffers[prior_index] / "training" / "summary.json"
        if not prior_summary.is_file():
            raise ValueError(f"round {prior_index:02d} is incomplete")
    candidate_path = root / "candidates.jsonl"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"round candidate file is missing: {candidate_path}")
    return PendingRound(
        index=index,
        root=root,
        buffer_path=root / "alignment_buffer.json",
        candidate_path=candidate_path,
        training_root=root / "training",
    )


def resolve_parent_checkpoint(
    pending: PendingRound,
    *,
    experiment_id: str,
    explicit_path: Path | None = None,
    explicit_sha256: str | None = None,
) -> ParentCheckpoint:
    """Resolve the released p0 parent or the preceding gated child checkpoint."""
    if (explicit_path is None) != (explicit_sha256 is None):
        raise ValueError(
            "--parent-checkpoint and --parent-checkpoint-sha256 must be supplied together"
        )
    if pending.index == 0:
        if explicit_path is None:
            return ParentCheckpoint("released_pepdfm", None, None, None, None)
        observed = sha256_file(explicit_path)
        if observed != str(explicit_sha256).lower():
            raise ValueError("explicit initial parent checkpoint hash differs")
        return ParentCheckpoint(
            "approved_external_checkpoint",
            explicit_path.resolve(),
            observed,
            None,
            None,
        )
    prior_root = pending.root.parent / f"round-{pending.index - 1:02d}" / "training"
    summary_path = prior_root / "summary.json"
    summary = read_json(summary_path)
    if summary.get("experiment_id") != experiment_id:
        raise ValueError("previous-round summary experiment mismatch")
    if int(summary.get("round_index", -1)) != pending.index - 1:
        raise ValueError("previous-round summary index mismatch")
    if summary.get("checkpoint_gate", {}).get("all_conditions_met") is not True:
        raise PermissionError("previous child checkpoint failed its registered gates")
    checkpoint_path = (prior_root / "checkpoints" / "child.pt").resolve()
    expected_hash = str(summary.get("checkpoint_sha256", {}).get("child", ""))
    if sha256_file(checkpoint_path) != expected_hash:
        raise ValueError("previous child checkpoint does not match its summary")
    if explicit_path is not None:
        if explicit_path.resolve() != checkpoint_path:
            raise ValueError("explicit parent checkpoint differs from the chained path")
        if str(explicit_sha256).lower() != expected_hash:
            raise ValueError("explicit parent checkpoint hash differs from the chain")
    return ParentCheckpoint(
        kind="previous_round_child",
        path=checkpoint_path,
        expected_sha256=expected_hash,
        source_round_index=pending.index - 1,
        source_summary_path=summary_path.resolve(),
    )


def _registered_lengths(config: Mapping[str, Any]) -> frozenset[int]:
    sampling = config["fresh_sampling"]
    mixture = sampling.get("length_counts_per_arm")
    if mixture is None:
        return frozenset({int(sampling["length"])})
    if not isinstance(mixture, Mapping) or not mixture:
        raise TypeError(
            "fresh_sampling.length_counts_per_arm must be a nonempty object"
        )
    return frozenset((int(length) for length in mixture))


def _candidate_rows(
    path: Path, *, expected_lengths: frozenset[int]
) -> list[dict[str, str]]:
    rows = read_jsonl(path)
    normalized: list[dict[str, str]] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        sequence = str(row.get("sequence", "")).strip().upper()
        if not candidate_id or not sequence:
            raise ValueError(
                "candidate rows require non-empty candidate_id and sequence"
            )
        encoded = encode_peptide(sequence)
        if len(encoded) - 2 not in expected_lengths:
            raise ValueError(
                "candidate sequence length is outside the registered mixture"
            )
        normalized.append({"candidate_id": candidate_id, "sequence": sequence})
    ids = [row["candidate_id"] for row in normalized]
    sequences = [row["sequence"] for row in normalized]
    if len(ids) != len(set(ids)) or len(sequences) != len(set(sequences)):
        raise ValueError("candidate IDs and sequences must be unique within a round")
    return normalized


def _validated_label_rows(
    rows: Any, *, label: str, candidate_by_id: Mapping[str, str]
) -> tuple[str, ...]:
    if not isinstance(rows, list):
        raise TypeError(f"stable_{label} must be a list")
    sequences: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError(f"stable_{label} rows must be objects")
        candidate_id = str(row.get("candidate_id", "")).strip()
        sequence = str(row.get("sequence", "")).strip().upper()
        if candidate_id in seen_ids:
            raise ValueError(f"stable_{label} contains duplicate candidate IDs")
        seen_ids.add(candidate_id)
        if candidate_by_id.get(candidate_id) != sequence:
            raise ValueError(
                f"stable_{label} row does not exactly match the round candidate pool"
            )
        sequences.append(sequence)
    return tuple(sequences)


def _repeat_agreement_floor(discovery: Mapping[str, Any]) -> float:
    """Resolve the registered two-pass agreement threshold across protocols.

    Earlier moving-frontier configurations call this field
    ``repeat_agreement_floor``.  The geometric V13 protocol registers the same
    categorical quantity as ``minimum_exact_agreement``.  Accept either name,
    reject an omitted threshold, and reject conflicting aliases so recovery
    cannot silently change the scientific gate.
    """
    values: list[tuple[str, float]] = []
    for key in ("repeat_agreement_floor", "minimum_exact_agreement"):
        if key not in discovery:
            continue
        try:
            value = float(discovery[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a finite fraction") from error
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be a finite fraction")
        values.append((key, value))
    if not values:
        raise ValueError(
            "curriculum_discovery must register a repeat-agreement threshold"
        )
    if len(values) == 2 and (
        not math.isclose(values[0][1], values[1][1], rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("repeat-agreement threshold aliases conflict")
    return values[0][1]


def validate_round_buffer(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    pending: PendingRound,
    parent: ParentCheckpoint,
) -> dict[str, Any]:
    buffer = read_json(pending.buffer_path)
    if buffer.get("experiment_id") != config["experiment_id"]:
        raise ValueError("alignment-buffer experiment mismatch")
    if int(buffer.get("round_index", -1)) != pending.index:
        raise ValueError("alignment-buffer round index mismatch")
    candidate_hash = sha256_file(pending.candidate_path)
    if buffer.get("candidate_pool_sha256") != candidate_hash:
        raise ValueError("alignment-buffer candidate-pool provenance mismatch")
    sampling = config["fresh_sampling"]
    candidates = _candidate_rows(
        pending.candidate_path, expected_lengths=_registered_lengths(config)
    )
    expected_count = int(config["curriculum_discovery"]["sample_size"])
    if len(candidates) != expected_count:
        raise ValueError(
            f"round candidate count {len(candidates)} differs from registered {expected_count}"
        )
    candidate_by_id = {row["candidate_id"]: row["sequence"] for row in candidates}
    positives = _validated_label_rows(
        buffer.get("stable_positive"), label="positive", candidate_by_id=candidate_by_id
    )
    negatives = _validated_label_rows(
        buffer.get("stable_negative"), label="negative", candidate_by_id=candidate_by_id
    )
    if set(positives).intersection(negatives):
        raise ValueError("stable positive and negative buffers overlap")
    discovery = config["curriculum_discovery"]
    if len(positives) < int(discovery["minimum_stable_positive_count"]):
        raise ValueError("stable positive buffer is below its registered gate")
    if len(negatives) < int(discovery["minimum_stable_negative_count"]):
        raise ValueError("stable negative buffer is below its registered gate")
    label_passes = int(discovery.get("training_label_passes", 2))
    if label_passes not in (1, 2):
        raise ValueError("training_label_passes must be one or two")
    stability = buffer.get("repeat_label_stability")
    if label_passes == 2:
        agreement_floor = _repeat_agreement_floor(discovery)
        agreement = (
            float(stability.get("agreement_fraction", -1.0))
            if isinstance(stability, Mapping)
            else -1.0
        )
        if (
            not math.isfinite(agreement)
            or not 0.0 <= agreement <= 1.0
            or agreement < agreement_floor
        ):
            raise ValueError("alignment buffer failed repeat-label stability")
    else:
        provenance = buffer.get("label_provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("pass_count") != 1
            or provenance.get("single_pass_labels") is not True
        ):
            raise ValueError("single-pass training lacks explicit label provenance")
    kind = str(config["fine_tuning"]["kind"])
    if buffer.get("negative_sequences_used_for_dpo_pairs") is not True:
        raise ValueError("flow-DPO buffer must authorize negative pair members")
    goal = buffer.get("training_goal")
    if not isinstance(goal, Mapping) or not str(goal.get("text", "")).strip():
        raise ValueError("training_goal must be a non-empty serialized goal")
    positive_train, positive_holdout = (tuple(sorted(positives)), ())
    negative_train, negative_holdout = (tuple(sorted(negatives)), ())
    parent_value = {
        "kind": parent.kind,
        "path": None if parent.path is None else str(parent.path),
        "expected_sha256": parent.expected_sha256,
        "source_round_index": parent.source_round_index,
        "source_summary_path": None
        if parent.source_summary_path is None
        else str(parent.source_summary_path),
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "round_index": pending.index,
        "generator_transition": str(
            buffer.get(
                "generator_transition", f"p{pending.index}->p{pending.index + 1}"
            )
        ),
        "config_sha256": sha256_file(config_path),
        "alignment_buffer_sha256": sha256_file(pending.buffer_path),
        "candidate_pool_path": str(pending.candidate_path),
        "candidate_pool_sha256": candidate_hash,
        "candidate_count": len(candidates),
        "training_goal": goal,
        "positive_train": positive_train,
        "positive_holdout": positive_holdout,
        "negative_train": negative_train,
        "negative_train_evaluation_only": (),
        "negative_holdout": negative_holdout,
        "abstentions_and_disagreements_excluded": True,
        "negative_gradient_weight": 0.0,
        "preference_pairing": "full_cartesian_stable_positive_x_stable_negative",
        "parent_checkpoint": parent_value,
    }


def cumulative_training_inputs(
    *,
    pending: PendingRound,
    validated: Mapping[str, Any],
    expected_lengths: frozenset[int],
    strict_fixed_goal_provenance: bool = False,
    strict_experiment_id: str | None = None,
    strict_goal_id: str | None = None,
    strict_training_goal: Mapping[str, Any] | None = None,
    contiguous_same_goal_only: bool = False,
) -> dict[str, Any]:
    """Combine stable labels through the current same-goal segment.

    Cross-round sequence conflicts are excluded from both classes.  This keeps
    the cumulative arm deterministic while preserving the current
    round's semantic goal as the evaluation frontier.  The legacy default
    combines all prior rounds; ``contiguous_same_goal_only`` resets replay at
    the most recent semantic-goal transition.
    """
    current_buffer = read_json(pending.buffer_path)
    expected_experiment_id = strict_experiment_id or str(
        current_buffer.get("experiment_id", "")
    )
    expected_goal_id = strict_goal_id or str(current_buffer.get("training_goal_id", ""))
    expected_goal = strict_training_goal or current_buffer.get("training_goal")
    if contiguous_same_goal_only and (
        not expected_goal_id or not isinstance(expected_goal, Mapping)
    ):
        raise ValueError("current cumulative buffer omitted its semantic goal")
    if strict_fixed_goal_provenance and (
        not expected_experiment_id
        or not expected_goal_id
        or (not isinstance(expected_goal, Mapping))
        or (current_buffer.get("experiment_id") != expected_experiment_id)
        or (current_buffer.get("training_goal_id") != expected_goal_id)
        or (current_buffer.get("training_goal") != expected_goal)
    ):
        raise ValueError("current cumulative buffer fixed-goal provenance differs")
    positives = set((str(value) for value in validated["positive_train"]))
    negatives = set((str(value) for value in validated["negative_train"]))
    provenance = [
        {
            "round_index": pending.index,
            "alignment_buffer_sha256": str(validated["alignment_buffer_sha256"]),
            "candidate_pool_sha256": str(validated["candidate_pool_sha256"]),
            "positive_count": len(positives),
            "negative_count": len(negatives),
        }
    ]
    prior_indices = list(range(pending.index))
    if contiguous_same_goal_only:
        prior_indices = list(
            contiguous_same_goal_prior_indices(
                current_index=pending.index,
                current_goal_id=expected_goal_id,
                current_goal=expected_goal,
                load_buffer=lambda index: read_json(
                    pending.root.parent / f"round-{index:02d}" / "alignment_buffer.json"
                ),
            )
        )
    for index in prior_indices:
        root = pending.root.parent / f"round-{index:02d}"
        candidate_path = root / "candidates.jsonl"
        buffer_path = root / "alignment_buffer.json"
        buffer = read_json(buffer_path)
        candidate_hash = sha256_file(candidate_path)
        if buffer.get("candidate_pool_sha256") != candidate_hash:
            raise ValueError("prior alignment-buffer candidate provenance mismatch")
        rows = _candidate_rows(candidate_path, expected_lengths=expected_lengths)
        candidate_by_id = {row["candidate_id"]: row["sequence"] for row in rows}
        if strict_fixed_goal_provenance:
            source = buffer.get("candidate_source")
            if (
                buffer.get("experiment_id") != expected_experiment_id
                or buffer.get("round_index") != index
                or buffer.get("generator_transition") != f"p{index}->p{index + 1}"
                or (buffer.get("training_goal_id") != expected_goal_id)
                or (buffer.get("training_goal") != expected_goal)
                or (
                    set(buffer.get("construction_candidate_ids", ()))
                    != set(candidate_by_id)
                )
                or (not isinstance(source, Mapping))
                or (source.get("kind") != "immutable_working_cohort")
                or (source.get("source_generator_id") != f"p{index}")
                or (buffer.get("abstentions_and_disagreements_excluded") is not True)
                or (buffer.get("negative_sequences_used_for_dpo_pairs") is not True)
            ):
                raise ValueError(
                    "prior cumulative buffer fixed-goal provenance differs"
                )
            manifest_path = Path(str(source.get("split_manifest_path", ""))).resolve()
            if not manifest_path.is_file() or sha256_file(manifest_path) != source.get(
                "split_manifest_sha256"
            ):
                raise ValueError("prior cumulative buffer distribution binding differs")
            manifest = read_json(manifest_path)
            source_artifact = manifest.get("source_artifact")
            if manifest.get("distribution_id") != f"p{index}" or not isinstance(
                source_artifact, Mapping
            ):
                raise ValueError("prior cumulative distribution identity differs")
            source_path = Path(str(source_artifact.get("path", ""))).resolve()
            if (
                not source_path.is_file()
                or sha256_file(source_path) != source_artifact.get("sha256")
                or {
                    row["candidate_id"]: row["sequence"]
                    for row in _candidate_rows(
                        source_path, expected_lengths=expected_lengths
                    )
                }
                != candidate_by_id
            ):
                raise ValueError(
                    "prior cumulative cohort differs from its distribution"
                )
        prior_positive = set(
            _validated_label_rows(
                buffer.get("stable_positive"),
                label="positive",
                candidate_by_id=candidate_by_id,
            )
        )
        prior_negative = set(
            _validated_label_rows(
                buffer.get("stable_negative"),
                label="negative",
                candidate_by_id=candidate_by_id,
            )
        )
        if strict_fixed_goal_provenance:
            stability = buffer.get("repeat_label_stability")
            if (
                not isinstance(stability, Mapping)
                or set(stability.get("stable_positive_ids", ()))
                != {
                    str(row.get("candidate_id", ""))
                    for row in buffer.get("stable_positive", ())
                }
                or set(stability.get("stable_negative_ids", ()))
                != {
                    str(row.get("candidate_id", ""))
                    for row in buffer.get("stable_negative", ())
                }
                or (stability.get("passed") is not True)
            ):
                raise ValueError("prior cumulative stable-label provenance differs")
            validated_path = root / "training" / "validated_inputs.json"
            summary_path = root / "training" / "summary.json"
            if not validated_path.is_file() or not summary_path.is_file():
                raise ValueError("prior cumulative training receipt is missing")
            prior_validated = read_json(validated_path)
            prior_summary = read_json(summary_path)
            summary_validated = dict(prior_summary.get("validated_inputs", {}))
            summary_validated.pop("execution_authorization", None)
            if (
                prior_validated != summary_validated
                or prior_validated.get("experiment_id") != expected_experiment_id
                or prior_validated.get("round_index") != index
                or (
                    prior_validated.get("generator_transition")
                    != f"p{index}->p{index + 1}"
                )
                or (
                    prior_validated.get("alignment_buffer_sha256")
                    != sha256_file(buffer_path)
                )
                or (prior_validated.get("candidate_pool_sha256") != candidate_hash)
                or (prior_validated.get("training_goal") != expected_goal)
                or (prior_summary.get("experiment_id") != expected_experiment_id)
                or (prior_summary.get("round_index") != index)
                or (
                    prior_summary.get("generator_transition")
                    != f"p{index}->p{index + 1}"
                )
                or (
                    prior_summary.get("checkpoint_gate", {}).get("all_conditions_met")
                    is not True
                )
            ):
                raise ValueError("prior cumulative training receipt differs")
        positives.update(prior_positive)
        negatives.update(prior_negative)
        provenance.append(
            {
                "round_index": index,
                "alignment_buffer_sha256": sha256_file(buffer_path),
                "candidate_pool_sha256": candidate_hash,
                "positive_count": len(prior_positive),
                "negative_count": len(prior_negative),
            }
        )
    conflicts = tuple(sorted(positives.intersection(negatives)))
    positives.difference_update(conflicts)
    negatives.difference_update(conflicts)
    result = dict(validated)
    result["positive_train"] = tuple(sorted(positives))
    result["negative_train"] = tuple(sorted(negatives))
    result["buffer_ablation"] = {
        "mode": "cumulative",
        "source_buffers": sorted(provenance, key=lambda value: value["round_index"]),
        "cross_round_conflicts_excluded": conflicts,
        "positive_count": len(positives),
        "negative_count": len(negatives),
    }
    return result


def sample_rows_registered(
    model: PepDfmCheckpoint, sampling: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Sample either the legacy fixed length or a balanced length mixture."""
    mixture = sampling.get("length_counts_per_arm")
    if mixture is None:
        return sample_rows(model, dict(sampling))
    if "length" in sampling:
        raise ValueError("mixed fresh sampling must not also specify length")
    counts = {int(length): int(count) for length, count in mixture.items()}
    if any((length <= 0 or count <= 0 for length, count in counts.items())):
        raise ValueError("mixed fresh-sampling lengths/counts must be positive")
    if sum(counts.values()) != int(sampling["num_samples"]):
        raise ValueError("length_counts_per_arm must sum to num_samples")
    by_length: dict[int, list[dict[str, Any]]] = {}
    base_seed = int(sampling["seed"])
    for length, count in sorted(counts.items()):
        values = dict(sampling)
        values.pop("length_counts_per_arm")
        values["num_samples"] = count
        values["length"] = length
        values["seed"] = base_seed + length
        values["batch_size"] = min(int(values["batch_size"]), count)
        by_length[length] = sample_rows(model, values)
    rows: list[dict[str, Any]] = []
    maximum = max(counts.values())
    for within_length_index in range(maximum):
        for length in sorted(counts):
            group = by_length[length]
            if within_length_index >= len(group):
                continue
            row = dict(group[within_length_index])
            row["attempt_index"] = len(rows)
            row["within_length_sample_index"] = int(row["sample_index"])
            row["length"] = length
            rows.append(row)
    return rows


def _load_checkpoint_state(path: Path, model: PepDfmCheckpoint) -> str:
    """Return the state hash of an existing safe state-dict checkpoint."""
    observed = sha256_file(path)
    checker = model.clone()
    loaded = checker.load_checkpoint(path, expected_sha256=observed)
    return loaded.state_dict_sha256


def immutable_save_checkpoint(model: PepDfmCheckpoint, path: Path) -> dict[str, str]:
    """Create a checkpoint once or accept a semantically identical artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_state_hash = sha256_state_dict(model.model)
    if path.exists():
        observed_state_hash = _load_checkpoint_state(path, model)
        if observed_state_hash != expected_state_hash:
            raise FileExistsError(f"immutable checkpoint differs: {path}")
        return {
            "file_sha256": sha256_file(path),
            "state_dict_sha256": observed_state_hash,
        }
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        model.save_checkpoint(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            observed_state_hash = _load_checkpoint_state(path, model)
            if observed_state_hash != expected_state_hash:
                raise FileExistsError(f"immutable checkpoint differs: {path}")
        return {
            "file_sha256": sha256_file(path),
            "state_dict_sha256": _load_checkpoint_state(path, model),
        }
    finally:
        temporary.unlink(missing_ok=True)


def _sequence_collection_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_training_contract(
    *,
    training_root: Path,
    positive_train: Sequence[str],
    negative_train: Sequence[str],
    train_config: FlowDPOConfig,
    pair_manifest_path: Path | None = None,
) -> dict[str, Any]:
    validated_path = training_root / "validated_inputs.json"
    if not validated_path.is_file():
        raise FileNotFoundError(
            "checkpoint creation requires immutable validated_inputs.json"
        )
    contract = {
        "validated_inputs_sha256": sha256_file(validated_path),
        "positive_train_sha256": _sequence_collection_sha256(positive_train),
        "negative_train_sha256": _sequence_collection_sha256(negative_train),
        "positive_count": len(positive_train),
        "negative_count": len(negative_train),
        "train_config": asdict(train_config),
    }
    if pair_manifest_path is None or not pair_manifest_path.is_file():
        raise FileNotFoundError("flow-DPO requires its explicit pair manifest")
    contract["pair_manifest"] = {
        "path": str(pair_manifest_path.resolve()),
        "sha256": sha256_file(pair_manifest_path),
    }
    return contract


def _checkpoint_stage_value(
    *,
    contract: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
    trace: Mapping[str, Any] | None,
    creator_authorization: Mapping[str, Any] | None,
    recovery_authorization: Mapping[str, Any] | None,
    recovery_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "training_contract": dict(contract),
        "checkpoint_artifacts": {
            name: dict(value) for name, value in artifacts.items()
        },
        "training_trace": None if trace is None else dict(trace),
        "creator_execution_authorization": None
        if creator_authorization is None
        else dict(creator_authorization),
        "recovery_execution_authorization": None
        if recovery_authorization is None
        else dict(recovery_authorization),
        "recovery_kind": recovery_kind,
    }


def _validate_checkpoint_stage(
    stage: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
) -> None:
    if stage.get("schema_version") != 1 or stage.get("status") != "complete":
        raise ValueError("checkpoint stage receipt is not complete")
    if stage.get("training_contract") != dict(contract):
        raise FileExistsError("checkpoint stage training contract differs")
    expected_artifacts = {name: dict(value) for name, value in artifacts.items()}
    if stage.get("checkpoint_artifacts") != expected_artifacts:
        raise FileExistsError("checkpoint stage artifact hashes differ")


def _validate_flow_dpo_trace(
    trace: Mapping[str, Any] | None,
    *,
    contract: Mapping[str, Any],
    parent_state_dict_sha256: str,
    child_state_dict_sha256: str,
    pair_manifest_path: Path,
) -> None:
    """Reject incomplete or self-inconsistent DPO receipts before sampling."""
    if not isinstance(trace, Mapping):
        raise ValueError("flow-DPO checkpoint stage lacks its training trace")
    train_config = contract.get("train_config")
    pair_binding = contract.get("pair_manifest")
    if not isinstance(train_config, Mapping) or not isinstance(pair_binding, Mapping):
        raise ValueError("flow-DPO checkpoint training contract is incomplete")
    pair_count = int(contract["positive_count"]) * int(contract["negative_count"])
    expected_slots = int(train_config["steps"]) * int(train_config["batch_size"])
    exact = {
        "objective": FLOW_DPO_KIND,
        "positive_count": int(contract["positive_count"]),
        "negative_count": int(contract["negative_count"]),
        "pair_count": pair_count,
        "pair_manifest_path": str(pair_manifest_path.resolve()),
        "pair_manifest_sha256": pair_binding.get("sha256"),
        "pair_slots_consumed": expected_slots,
        "config": dict(train_config),
        "checkpoint_sha256_before": parent_state_dict_sha256,
        "checkpoint_sha256_after": child_state_dict_sha256,
        "reference_checkpoint_sha256_before": parent_state_dict_sha256,
        "reference_checkpoint_sha256_after": parent_state_dict_sha256,
        "all_pairs_visited": True,
        "reference_checkpoint_unchanged": True,
        "flow_dpo_surrogate_not_exact_sequence_log_likelihood": True,
        "training_batch_diagnostics_are_not_fixed_set_pre_post_metrics": True,
    }
    if any((trace.get(name) != expected for name, expected in exact.items())):
        raise ValueError("flow-DPO training trace differs from checkpoint contract")
    if (
        int(trace.get("minimum_pair_visit_count", 0)) < 1
        or int(trace.get("maximum_pair_visit_count", 0))
        < int(trace.get("minimum_pair_visit_count", 0))
        or int(trace.get("complete_pair_epochs", -1)) != expected_slots // pair_count
    ):
        raise ValueError("flow-DPO training trace has invalid pair coverage")
    required_batch_diagnostics = {
        "first_batch_dpo_loss",
        "last_batch_dpo_loss",
        "minimum_training_batch_dpo_loss",
        "maximum_training_batch_dpo_loss",
        "first_batch_preference_accuracy",
        "last_batch_preference_accuracy",
        "last_batch_mean_implicit_margin",
        "last_batch_mean_preferred_loss_change_vs_reference",
        "last_batch_mean_rejected_loss_change_vs_reference",
    }
    if any((name not in trace for name in required_batch_diagnostics)):
        raise ValueError("flow-DPO training trace lacks batch diagnostics")


def _validate_legacy_checkpoint_binding(
    binding: Mapping[str, Any] | None,
    *,
    round_index: int,
    contract: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, str]],
) -> None:
    if not isinstance(binding, Mapping):
        raise PermissionError(
            "a complete checkpoint pair without a stage receipt requires an explicit recovery binding"
        )
    expected = {
        "round_index": round_index,
        "validated_inputs_sha256": contract["validated_inputs_sha256"],
        "parent": dict(artifacts["parent"]),
        "child": dict(artifacts["child"]),
    }
    observed = {
        "round_index": binding.get("round_index"),
        "validated_inputs_sha256": binding.get("validated_inputs_sha256"),
        "parent": binding.get("parent"),
        "child": binding.get("child"),
    }
    if observed != expected:
        raise PermissionError("legacy checkpoint pair differs from recovery binding")


def _write_checkpoint_stage(path: Path, value: Mapping[str, Any]) -> None:
    immutable_write(
        path, json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )


def _atomic_save_fresh_checkpoint_bundle(
    *,
    parent: PepDfmCheckpoint,
    child: PepDfmCheckpoint,
    checkpoint_root: Path,
    contract: Mapping[str, Any],
    trace: Mapping[str, Any],
    execution_authorization: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    """Publish both checkpoints and their receipt with one directory rename."""
    if checkpoint_root.exists():
        raise FileExistsError(
            "fresh checkpoint destination already exists without a recoverable pair"
        )
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".checkpoints-stage-", dir=checkpoint_root.parent)
    )
    try:
        temporary_parent = temporary_root / "parent.pt"
        temporary_child = temporary_root / "child.pt"
        parent.save_checkpoint(temporary_parent)
        child.save_checkpoint(temporary_child)
        artifacts = {
            "parent": {
                "file_sha256": sha256_file(temporary_parent),
                "state_dict_sha256": _load_checkpoint_state(temporary_parent, parent),
            },
            "child": {
                "file_sha256": sha256_file(temporary_child),
                "state_dict_sha256": _load_checkpoint_state(temporary_child, child),
            },
        }
        stage_value = _checkpoint_stage_value(
            contract=contract,
            artifacts=artifacts,
            trace=trace,
            creator_authorization=execution_authorization,
            recovery_authorization=None,
            recovery_kind="fresh_training",
        )
        (temporary_root / "stage.json").write_text(
            json.dumps(stage_value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(temporary_root, checkpoint_root)
        return artifacts
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)


def train_or_recover_checkpoint_pair(
    *,
    parent: PepDfmCheckpoint,
    child: PepDfmCheckpoint,
    training_root: Path,
    positive_train: Sequence[str],
    negative_train: Sequence[str],
    train_config: FlowDPOConfig,
    pair_manifest_path: Path | None = None,
    round_index: int = 0,
    execution_authorization: Mapping[str, Any] | None = None,
    legacy_recovery_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Train once, or safely recover a complete checkpoint pair after a crash.

    Checkpoints are written before sampling and predictor evaluation. A process
    may therefore die after training but before ``summary.json``. On re-entry,
    verify the persisted parent, architecture-check and load the child, and
    continue without repeating optimization. New pairs carry a durable stage
    receipt. The one pre-receipt EXP-0077 pair is accepted only through an exact
    campaign binding. A parent-only legacy pair can be completed
    deterministically; a child-only pair is always rejected.
    """
    checkpoint_root = training_root / "checkpoints"
    parent_path = checkpoint_root / "parent.pt"
    child_path = checkpoint_root / "child.pt"
    stage_path = checkpoint_root / "stage.json"
    if child_path.exists() and (not parent_path.exists()):
        raise FileExistsError(
            "child checkpoint exists without its parent; refusing ambiguous recovery"
        )
    contract = _checkpoint_training_contract(
        training_root=training_root,
        positive_train=positive_train,
        negative_train=negative_train,
        train_config=train_config,
        pair_manifest_path=pair_manifest_path,
    )
    parent_state = sha256_state_dict(parent.model)
    if parent_path.exists() and child_path.exists():
        persisted_parent_state = _load_checkpoint_state(parent_path, parent)
        if persisted_parent_state != parent_state:
            raise FileExistsError(
                "persisted parent checkpoint differs from the resolved parent"
            )
        child_file_sha256 = sha256_file(child_path)
        loaded_child = child.load_checkpoint(
            child_path, expected_sha256=child_file_sha256
        )
        if loaded_child.state_dict_sha256 == parent_state:
            raise ValueError("recovered child checkpoint is identical to its parent")
        artifacts = {
            "parent": {
                "file_sha256": sha256_file(parent_path),
                "state_dict_sha256": persisted_parent_state,
            },
            "child": {
                "file_sha256": child_file_sha256,
                "state_dict_sha256": loaded_child.state_dict_sha256,
            },
        }
        if stage_path.is_file():
            stage = read_json(stage_path)
            _validate_checkpoint_stage(stage, contract=contract, artifacts=artifacts)
            stored_trace = stage.get("training_trace")
            if pair_manifest_path is None:
                raise ValueError("flow-DPO recovery lacks its pair manifest")
            _validate_flow_dpo_trace(
                stored_trace if isinstance(stored_trace, Mapping) else None,
                contract=contract,
                parent_state_dict_sha256=persisted_parent_state,
                child_state_dict_sha256=loaded_child.state_dict_sha256,
                pair_manifest_path=pair_manifest_path,
            )
            trace = dict(stored_trace) if isinstance(stored_trace, Mapping) else {}
            original_trace_available = isinstance(stored_trace, Mapping)
            recovery_kind = "stage_receipt"
        else:
            raise PermissionError(
                "flow-DPO checkpoint recovery requires its complete stage trace"
            )
            _validate_legacy_checkpoint_binding(
                legacy_recovery_binding,
                round_index=round_index,
                contract=contract,
                artifacts=artifacts,
            )
            validated_value = read_json(training_root / "validated_inputs.json")
            creator_value = validated_value.get("execution_authorization")
            creator = creator_value if isinstance(creator_value, Mapping) else None
            _write_checkpoint_stage(
                stage_path,
                _checkpoint_stage_value(
                    contract=contract,
                    artifacts=artifacts,
                    trace=None,
                    creator_authorization=creator,
                    recovery_authorization=execution_authorization,
                    recovery_kind="legacy_exact_hash_binding",
                ),
            )
            trace = {}
            original_trace_available = False
            recovery_kind = "legacy_exact_hash_binding"
        common_recovery = {
            "positive_count": len(positive_train),
            "negative_count": len(negative_train),
            "config": asdict(train_config),
            "maximum_preclip_gradient_norm": trace.get("maximum_preclip_gradient_norm"),
            "checkpoint_sha256_before": parent_state,
            "checkpoint_sha256_after": loaded_child.state_dict_sha256,
            "recovered_existing_checkpoint_pair": True,
            "fine_tuning_reexecuted": False,
            "checkpoint_recovery_kind": recovery_kind,
            "recovery_limitation": None
            if original_trace_available
            else "The legacy process committed checkpoints before a stage receipt; exact checkpoint states and the immutable training contract are verified, but its in-memory loss trace is unavailable.",
        }
        common_recovery.update(
            objective=FLOW_DPO_KIND,
            pair_manifest_path=str(pair_manifest_path.resolve()),
            pair_manifest_sha256=sha256_file(pair_manifest_path),
            pair_count=len(positive_train) * len(negative_train),
            all_pairs_visited=trace.get("all_pairs_visited"),
            first_batch_dpo_loss=trace.get("first_batch_dpo_loss"),
            last_batch_dpo_loss=trace.get("last_batch_dpo_loss"),
            reference_checkpoint_unchanged=trace.get("reference_checkpoint_unchanged"),
        )
        trace.update(common_recovery)
        return (trace, artifacts)
    completing_parent_only = parent_path.exists()
    if completing_parent_only:
        persisted_parent_state = _load_checkpoint_state(parent_path, parent)
        if persisted_parent_state != parent_state:
            raise FileExistsError(
                "persisted parent checkpoint differs from the resolved parent"
            )
    if pair_manifest_path is None:
        raise ValueError("flow-DPO requires an explicit pair manifest")
    trace = fine_tune_flow_dpo(
        policy=child,
        reference=parent,
        positive_sequences=positive_train,
        negative_sequences=negative_train,
        pair_manifest_path=pair_manifest_path,
        config=train_config,
    )
    _validate_flow_dpo_trace(
        trace,
        contract=contract,
        parent_state_dict_sha256=parent_state,
        child_state_dict_sha256=sha256_state_dict(child.model),
        pair_manifest_path=pair_manifest_path,
    )
    if completing_parent_only:
        artifacts = {
            "parent": immutable_save_checkpoint(parent, parent_path),
            "child": immutable_save_checkpoint(child, child_path),
        }
        _write_checkpoint_stage(
            stage_path,
            _checkpoint_stage_value(
                contract=contract,
                artifacts=artifacts,
                trace=trace,
                creator_authorization=execution_authorization,
                recovery_authorization=None,
                recovery_kind="deterministic_parent_only_completion",
            ),
        )
    else:
        artifacts = _atomic_save_fresh_checkpoint_bundle(
            parent=parent,
            child=child,
            checkpoint_root=checkpoint_root,
            contract=contract,
            trace=trace,
            execution_authorization=execution_authorization,
        )
    trace["recovered_existing_checkpoint_pair"] = False
    trace["fine_tuning_reexecuted"] = True
    trace["checkpoint_recovery_kind"] = (
        "deterministic_parent_only_completion"
        if completing_parent_only
        else "fresh_training"
    )
    return (trace, artifacts)


def _write_sample_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text = "".join(
        (json.dumps(dict(row), allow_nan=False, sort_keys=True) + "\n" for row in rows)
    )
    immutable_write(path, text)
    return {"path": str(path), "sha256": sha256_file(path), "row_count": len(rows)}


def _load_sample_rows_exact(path: Path, *, expected_count: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(
            f"sample artifact {path} has {len(rows)} rows, expected {expected_count}"
        )
    return rows


def sample_or_recover_raw_rows(
    *,
    models: Mapping[str, PepDfmCheckpoint],
    sampling: Mapping[str, Any],
    training_root: Path,
    checkpoint_artifacts: Mapping[str, Mapping[str, str]],
    precomputed_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Persist each raw sampling arm before any predictor annotation.

    ``precomputed_rows_by_arm`` is used by V13 to reuse the already immutable
    100-peptide current-generator cohort as the parent diagnostic arm.  This
    avoids drawing a second parent cohort while preserving the historical V12
    behavior when the argument is omitted.
    """
    if set(models) != {"parent", "child"}:
        raise ValueError("raw sampling requires exactly parent and child models")
    precomputed = {
        str(arm): [dict(row) for row in rows]
        for arm, rows in dict(precomputed_rows_by_arm or {}).items()
    }
    if not set(precomputed).issubset({"parent", "child"}):
        raise ValueError("precomputed raw rows contain an unsupported arm")
    expected_count = int(sampling["num_samples"])
    stage_path = training_root / "raw_sampling" / "stage.json"
    paths = {
        arm: training_root / "raw_sampling" / f"{arm}.jsonl"
        for arm in ("parent", "child")
    }
    if stage_path.is_file():
        stage = read_json(stage_path)
        if stage.get("sampling_config") != dict(sampling):
            raise FileExistsError("raw-sampling stage config differs")
        expected_precomputed_hashes = {
            arm: hashlib.sha256(
                "".join(
                    (
                        json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
                        for row in rows
                    )
                ).encode("utf-8")
            ).hexdigest()
            for arm, rows in sorted(precomputed.items())
        }
        if stage.get("precomputed_arm_sha256", {}) != expected_precomputed_hashes:
            raise FileExistsError("raw-sampling precomputed-arm binding differs")
        expected_states = {
            arm: checkpoint_artifacts[arm]["state_dict_sha256"]
            for arm in ("parent", "child")
        }
        if stage.get("model_state_dict_sha256") != expected_states:
            raise FileExistsError("raw-sampling checkpoint binding differs")
        rows_by_arm: dict[str, list[dict[str, Any]]] = {}
        for arm, path in paths.items():
            record = stage.get("sample_files", {}).get(arm, {})
            if (
                record.get("sha256") != sha256_file(path)
                or int(record.get("row_count", -1)) != expected_count
            ):
                raise FileExistsError(
                    "raw-sampling artifact differs from stage receipt"
                )
            rows_by_arm[arm] = _load_sample_rows_exact(
                path, expected_count=expected_count
            )
        return (rows_by_arm, stage)
    rows_by_arm = {}
    artifacts: dict[str, Any] = {}
    for arm, path in paths.items():
        regenerated = (
            [dict(row) for row in precomputed[arm]]
            if arm in precomputed
            else sample_rows_registered(models[arm], sampling)
        )
        if len(regenerated) != expected_count:
            raise ValueError("deterministic raw sampling returned the wrong row count")
        if path.is_file():
            orphan = _load_sample_rows_exact(path, expected_count=expected_count)
            if json.loads(json.dumps(orphan)) != json.loads(json.dumps(regenerated)):
                raise FileExistsError(
                    f"orphan raw-sampling artifact differs from regenerated {arm} arm"
                )
        rows_by_arm[arm] = regenerated
        artifacts[arm] = _write_sample_rows(path, regenerated)
    stage = {
        "schema_version": 1,
        "status": "complete",
        "sampling_config": dict(sampling),
        "model_state_dict_sha256": {
            arm: checkpoint_artifacts[arm]["state_dict_sha256"]
            for arm in ("parent", "child")
        },
        "precomputed_arm_sha256": {
            arm: hashlib.sha256(
                "".join(
                    (
                        json.dumps(row, allow_nan=False, sort_keys=True) + "\n"
                        for row in rows
                    )
                ).encode("utf-8")
            ).hexdigest()
            for arm, rows in sorted(precomputed.items())
        },
        "sample_files": artifacts,
    }
    immutable_write(
        stage_path, json.dumps(stage, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return (rows_by_arm, stage)


def all_prior_label_sequences(output_root: Path, *, through_round: int) -> set[str]:
    """Return every sequence exposed as a stable label through this round."""
    values: set[str] = set()
    for index in range(through_round + 1):
        path = output_root / "rounds" / f"round-{index:02d}" / "alignment_buffer.json"
        buffer = read_json(path)
        for field in ("stable_positive", "stable_negative"):
            rows = buffer.get(field)
            if not isinstance(rows, list):
                raise TypeError(f"{path} omitted {field}")
            for row in rows:
                if not isinstance(row, Mapping) or not str(row.get("sequence", "")):
                    raise ValueError(f"{path} contains an invalid stable label")
                values.add(str(row["sequence"]))
    return values
