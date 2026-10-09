#!/usr/bin/env python3
"""CPU-only terminal validator for the V13 geometric PepDFM campaign.

This program is intentionally separate from the controller.  It does not load
PepDFM, import torch, call a provider, or make a scientific decision.  It
reconstructs the categorical partitions from the two raw judgment passes and
checks the immutable hash chain that connects the frozen configuration,
working cohorts, all-label update receipts, terminal state, and final report.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

from domains.peptides.reachability import (
    audit_learnability,
    categorical_report_by_length,
)
from domains.peptides.audit import build_q_matrix_cell
from domains.peptides.length_policy import (
    load_pre_p0_length_policy_artifact,
)
from domains.peptides.types import PeptideCandidate


PROTOCOL_VERSION = "geometric-semantic-reachability-v13"
ANTICANCER_FINAL_GOAL = (
    "Prefer peptides that appear more promising as selective direct "
    "anticancer therapeutics: plausibly active against cancer cells "
    "while minimizing nonspecific toxicity toward healthy human cells."
)
FINAL_GOAL_ID = "F"
ROLE_NAMES = ("candidate_judge", "curriculum_controller", "independent_auditor")
STATE_RE = re.compile(r"snapshot-([0-9]{3})\.json")
DIST_RE = re.compile(r"p([0-9]+)")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
COUNTED_TRANSITIONS = frozenset(
    {"newly_reachable_smallest_tightening", "exact_final_newly_reachable"}
)
TRANSITION_COUNT_METRIC = "completed_geometric_semantic_transition_count"
EXACT_FINAL_PRIMARY_METRIC = (
    "exact_final_goal_newly_reachable_after_generator_updates"
)
FLOW_DPO_KIND = "all_cartesian_pairs_pepdfm_flow_dpo"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"non-empty JSONL required: {path}")
    return rows


def immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a validation receipt without replacing an existing artifact."""

    text = json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"immutable validation artifact differs: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(
                    f"immutable validation artifact differs: {path}"
                )
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _path(value: Any) -> Path:
    return Path(os.path.expandvars(str(value))).resolve()


def _verify_file(path: Path, expected: Any) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_hash = _require_sha256(expected, name=str(path))
    observed = sha256_file(path)
    if observed != expected_hash:
        raise ValueError(f"artifact SHA-256 differs: {path}")
    return observed


def _verify_binding(
    binding: Any,
    *,
    name: str,
    hash_key: str = "sha256",
) -> Path:
    if not isinstance(binding, Mapping):
        raise TypeError(f"{name} must be an artifact binding")
    path = _path(binding.get("path", ""))
    _verify_file(path, binding.get(hash_key))
    return path


def _length_counts(config: Mapping[str, Any]) -> dict[int, int]:
    raw = config.get("fresh_sampling", {}).get("length_counts_per_arm")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("frozen sampling length policy is missing")
    counts = {int(key): int(value) for key, value in raw.items()}
    if (
        not 2 <= len(counts) <= 4
        or sum(counts.values()) != 100
        or any(not 6 <= length <= 49 for length in counts)
        or any(not 10 <= count <= 60 for count in counts.values())
    ):
        raise ValueError("frozen campaign length policy is invalid")
    return dict(sorted(counts.items()))


def _validate_frozen_config(
    config_path: Path, config: Mapping[str, Any]
) -> tuple[dict[int, int], dict[str, Any]]:
    if config.get("prompt_protocol", {}).get("version") != PROTOCOL_VERSION:
        raise ValueError("terminal validator requires the V13 protocol")
    if (
        config.get("final_goal") != ANTICANCER_FINAL_GOAL
        or config.get("evaluation", {}).get("cardinal_scores_used") is not False
        or config.get("evaluation", {}).get(
            "independent_generated_validation_cohort"
        )
        is not False
        or config.get("evaluation", {}).get("working_cohort_size") != 100
        or config.get("prompt_protocol", {}).get(
            "cardinal_predictor_outputs_visible"
        )
        is not False
        or config.get("curriculum_loop", {}).get("llm_cardinal_scores_forbidden")
        is not True
    ):
        raise ValueError("V13 categorical one-cohort contract differs")
    tuning = config.get("fine_tuning")
    kind = tuning.get("kind") if isinstance(tuning, Mapping) else None
    if kind != FLOW_DPO_KIND:
        raise ValueError("unsupported anticancer fine-tuning contract")
    if kind == FLOW_DPO_KIND:
        expected_dpo = {
            "holdout_fraction": 0.0,
            "steps": 100,
            "batch_size": 32,
            "negative_alpha": 0.0,
            "beta": 2.6,
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
        if any(tuning.get(name) != value for name, value in expected_dpo.items()):
            raise ValueError("terminal flow-DPO contract differs")
    counts = _length_counts(config)
    policy_binding = config.get("campaign_length_policy")
    policy_path = _verify_binding(policy_binding, name="campaign length policy")
    policy = load_pre_p0_length_policy_artifact(policy_path)
    policy_counts = {
        row.length: row.count for row in policy.length_policy.allocations
    }
    if (
        policy_counts != counts
        or policy.final_goal != config.get("final_goal")
        or policy_binding.get("selected_before_p0") is not True
        or policy_binding.get("frozen_for_entire_campaign") is not True
        or {
            int(key): int(value)
            for key, value in dict(policy_binding.get("length_counts", {})).items()
        }
        != counts
    ):
        raise ValueError("length-policy artifact differs from frozen config")

    candidate_path = _verify_binding(
        config.get("candidate_pool"), name="candidate pool"
    )
    evidence_path = _verify_binding(
        config.get("fresh_sampling", {}).get("frozen_evidence_reference"),
        name="frozen p0 evidence reference",
    )
    if (
        config.get("anticancer_reference_context") is not None
        or config.get("prompt_protocol", {}).get(
            "labelled_reference_context_visible"
        )
        is not False
        or config.get("prompt_protocol", {}).get(
            "external_peptide_dataset_visible"
        )
        is not False
    ):
        raise ValueError("an external peptide dataset crossed the V13 boundary")

    manifest_path = config_path.with_name("frozen_config_manifest.json")
    manifest = read_json(manifest_path)
    expected = {
        "template_sha256": manifest.get("template_sha256"),
        "frozen_config_path": str(config_path.resolve()),
        "frozen_config_sha256": sha256_file(config_path),
        "candidate_pool_sha256": sha256_file(candidate_path),
        "frozen_evidence_reference_sha256": sha256_file(evidence_path),
        "campaign_length_policy_sha256": sha256_file(policy_path),
        "external_dataset_context_visible": False,
        "length_counts": {str(key): value for key, value in counts.items()},
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen config manifest does not bind its runtime inputs")
    template_path = _path(manifest.get("template_path", ""))
    _verify_file(template_path, manifest.get("template_sha256"))
    if manifest.get("transformation") != "V13 bounded dynamic-artifact hash binding only":
        raise ValueError("unregistered frozen-config transformation")
    return counts, manifest


def _state_snapshots(output_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    root = output_root / "state"
    paths = sorted(root.glob("snapshot-*.json")) if root.exists() else []
    if not paths:
        raise ValueError("campaign has no immutable state snapshots")
    result: list[tuple[Path, dict[str, Any]]] = []
    prior_hash: str | None = None
    prior: dict[str, Any] | None = None
    for index, path in enumerate(paths):
        match = STATE_RE.fullmatch(path.name)
        if match is None or int(match.group(1)) != index:
            raise ValueError("state snapshots are not contiguous")
        value = read_json(path)
        if (
            value.get("snapshot_index") != index
            or value.get("prior_snapshot_sha256") != prior_hash
        ):
            raise ValueError("state snapshot hash chain differs")
        for key in (
            "goal_ids",
            "distribution_ids",
            "updates",
            "accepted_transitions",
        ):
            if not isinstance(value.get(key), list):
                raise ValueError(f"state snapshot omitted list field {key}")
            if prior is not None and value[key][: len(prior[key])] != prior[key]:
                raise ValueError(f"state snapshot rewrote append-only field {key}")
        if prior is not None and int(value.get("total_updates", -1)) < int(
            prior.get("total_updates", -1)
        ):
            raise ValueError("state total_updates moved backwards")
        result.append((path, value))
        prior_hash = sha256_file(path)
        prior = value
    return result


def _candidate_rows(
    path: Path, *, counts: Mapping[int, int]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows = read_jsonl(path)
    if len(rows) != 100:
        raise ValueError(f"working cohort must contain exactly 100 rows: {path}")
    ids: list[str] = []
    sequences: list[str] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        sequence = str(row.get("sequence", "")).strip().upper()
        if (
            not candidate_id
            or not sequence
            or not set(sequence).issubset(ALPHABET)
            or row.get("is_valid", True) is not True
            or not isinstance(row.get("developability_evidence"), Mapping)
        ):
            raise ValueError(f"invalid working-cohort candidate: {path}")
        if row.get("length") is not None and int(row["length"]) != len(sequence):
            raise ValueError(f"stored peptide length differs: {path}")
        ids.append(candidate_id)
        sequences.append(sequence)
    if len(set(ids)) != 100 or len(set(sequences)) != 100:
        raise ValueError(f"working-cohort IDs and sequences must be unique: {path}")
    if Counter(map(len, sequences)) != Counter(counts):
        raise ValueError(f"working cohort differs from frozen length policy: {path}")
    return rows, dict(zip(ids, sequences))


def _validate_distributions(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    state: Mapping[str, Any],
    counts: Mapping[int, int],
) -> dict[str, dict[str, Any]]:
    distribution_ids = [str(value) for value in state["distribution_ids"]]
    expected_ids = [f"p{index}" for index in range(int(state["total_updates"]) + 1)]
    if distribution_ids != expected_ids:
        raise ValueError("state distribution topology differs from completed updates")
    manifest_paths = {
        path.parent.name: path
        for path in output_root.glob("distributions/*/distribution_split_manifest.json")
    }
    if set(manifest_paths) != set(distribution_ids):
        raise ValueError("unexpected or missing generated distribution manifest")

    values: dict[str, dict[str, Any]] = {}
    all_sequences: set[str] = set()
    for distribution_id in distribution_ids:
        manifest_path = manifest_paths[distribution_id]
        if DIST_RE.fullmatch(distribution_id) is None:
            raise ValueError("non-generator distribution present in V13 campaign")
        manifest = read_json(manifest_path)
        if (
            manifest.get("protocol_version") != PROTOCOL_VERSION
            or manifest.get("distribution_id") != distribution_id
            or manifest.get("cohort_kind") != "single_immutable_working_cohort"
            or manifest.get("candidate_count") != 100
            or manifest.get("independent_generated_validation_cohort") is not False
            or {int(key): int(value) for key, value in dict(manifest.get("length_counts", {})).items()}
            != dict(counts)
        ):
            raise ValueError(f"working-distribution manifest differs: {manifest_path}")
        source = manifest.get("source_artifact")
        source_path = _verify_binding(source, name=f"{distribution_id} source")
        if source.get("row_count") != 100:
            raise ValueError(f"{distribution_id} source row count differs")
        rows, by_id = _candidate_rows(source_path, counts=counts)
        if manifest.get("candidate_id_sha256") != hashlib.sha256(
            canonical_json(list(by_id)).encode("utf-8")
        ).hexdigest():
            raise ValueError(f"{distribution_id} candidate-ID binding differs")
        overlap = all_sequences.intersection(by_id.values())
        if overlap:
            raise ValueError("working generator distributions overlap by sequence")
        all_sequences.update(by_id.values())
        values[distribution_id] = {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "source_path": source_path,
            "rows": rows,
            "by_id": by_id,
        }
    p0_source = values["p0"]["source_path"]
    if (
        p0_source != _path(config["candidate_pool"]["path"])
        or sha256_file(p0_source) != config["candidate_pool"]["sha256"]
    ):
        raise ValueError("p0 does not bind the frozen candidate pool")
    return values


def _labels(envelope: Mapping[str, Any]) -> tuple[dict[str, str], Mapping[str, Any]]:
    result = envelope.get("result")
    if not isinstance(result, Mapping) or not isinstance(result.get("judgments"), list):
        raise ValueError("judgment pass omitted structured results")
    labels: dict[str, str] = {}
    for row in result["judgments"]:
        if not isinstance(row, Mapping):
            raise ValueError("judgment row is malformed")
        candidate_id = str(row.get("candidate_id", ""))
        label = str(row.get("label", ""))
        if not candidate_id or label not in {"positive", "negative", "abstain"}:
            raise ValueError("judgment row has an invalid categorical label")
        if candidate_id in labels:
            raise ValueError("judgment pass contains duplicate candidate IDs")
        labels[candidate_id] = label
    if len(labels) != 100:
        raise ValueError("judgment pass must partition exactly 100 candidates")
    counts = Counter(labels.values())
    expected_counts = {
        "sample_count": 100,
        "positive_count": counts["positive"],
        "negative_count": counts["negative"],
        "abstain_count": counts["abstain"],
        "positive_fraction_all": counts["positive"] / 100,
        "abstain_fraction": counts["abstain"] / 100,
    }
    if result.get("counts") is not None and result.get("counts") != expected_counts:
        raise ValueError("judgment pass counts differ from its labels")
    return labels, result


def _validate_categorical_reports(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    distributions: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, str]], dict[tuple[str, str, str], dict[str, Any]]]:
    report_paths = sorted(output_root.glob("judgments/*/*/categorical_report.json"))
    if not report_paths:
        raise ValueError("terminal campaign contains no categorical judgments")
    records: list[dict[str, str]] = []
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for report_path in report_paths:
        root = report_path.parent
        distribution_id = root.parent.name
        if distribution_id not in distributions:
            raise ValueError("judgment names an unknown generator distribution")
        first_envelope = read_json(root / "pass-0.json")
        repeat_envelope = read_json(root / "pass-1.json")
        first, _ = _labels(first_envelope)
        repeat, _ = _labels(repeat_envelope)
        by_id = distributions[distribution_id]["by_id"]
        if set(first) != set(by_id) or set(repeat) != set(by_id):
            raise ValueError("judgment pair differs from its working cohort")
        goal_id = str(first_envelope.get("goal_id", ""))
        judgment_key = root.name
        if (
            repeat_envelope.get("goal_id") != goal_id
            or first_envelope.get("distribution_id") != distribution_id
            or repeat_envelope.get("distribution_id") != distribution_id
            or first_envelope.get("judgment_key") != judgment_key
            or repeat_envelope.get("judgment_key") != judgment_key
            or first_envelope.get("repeat_index") != 0
            or repeat_envelope.get("repeat_index") != 1
            or first_envelope.get("candidate_count") != 100
            or repeat_envelope.get("candidate_count") != 100
        ):
            raise ValueError("judgment envelope identity differs")
        cell = build_q_matrix_cell(
            generator_id=distribution_id,
            goal_id=goal_id,
            first_labels=first,
            repeat_labels=repeat,
            strata={candidate_id: str(len(sequence)) for candidate_id, sequence in by_id.items()},
        )
        stable_path = root / "stable_summary.json"
        if read_json(stable_path) != cell.as_mapping(include_outcomes=True):
            raise ValueError("stable categorical partition differs from raw passes")
        candidates = tuple(
            PeptideCandidate(candidate_id, sequence)
            for candidate_id, sequence in by_id.items()
        )
        expected_report = categorical_report_by_length(candidates=candidates, cell=cell)
        if read_json(report_path) != expected_report:
            raise ValueError("per-length categorical report differs from raw passes")
        key = (distribution_id, goal_id, judgment_key)
        if key in cells:
            raise ValueError("duplicate categorical-cell identity")
        cells[key] = {
            "cell": cell,
            "root": root,
            "first_result": first_envelope["result"],
            "repeat_result": repeat_envelope["result"],
        }
        records.append({"path": str(report_path.resolve()), "sha256": sha256_file(report_path)})
    return records, cells


def _average_pairwise_hamming(sequences: Sequence[str]) -> float | None:
    unique = sorted(set(sequences))
    by_length: dict[int, list[str]] = {}
    for sequence in unique:
        by_length.setdefault(len(sequence), []).append(sequence)
    total = 0.0
    pairs = 0
    for length, rows in sorted(by_length.items()):
        for left, right in itertools.combinations(rows, 2):
            total += sum(a != b for a, b in zip(left, right)) / length
            pairs += 1
    return None if pairs == 0 else total / pairs


def _amino_acid_distribution(sequences: Sequence[str]) -> dict[str, float]:
    counts = Counter("".join(sequences))
    total = sum(counts.values())
    return {
        residue: (counts[residue] / total if total else 0.0)
        for residue in sorted(ALPHABET)
    }


def _jensen_shannon(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    value = 0.0
    for key in sorted(set(left).union(right)):
        p = float(left.get(key, 0.0))
        q = float(right.get(key, 0.0))
        midpoint = 0.5 * (p + q)
        if p > 0:
            value += 0.5 * p * math.log2(p / midpoint)
        if q > 0:
            value += 0.5 * q * math.log2(q / midpoint)
    return value


def _same_number(observed: Any, expected: float) -> bool:
    return isinstance(observed, (int, float)) and not isinstance(observed, bool) and math.isclose(
        float(observed), expected, rel_tol=1e-12, abs_tol=1e-12
    )


def _validate_alignment_goal_binding(
    *, output_root: Path, buffer: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Bind one training buffer to the complete immutable goal artifact."""

    goal_id = str(buffer.get("training_goal_id", ""))
    if not goal_id:
        raise ValueError("alignment buffer omitted its training goal ID")
    goal_path = output_root / "goals" / f"{goal_id}.json"
    artifact = read_json(goal_path)
    goal = artifact.get("goal")
    if (
        artifact.get("schema_version") != 1
        or artifact.get("goal_id") != goal_id
        or not isinstance(goal, Mapping)
    ):
        raise ValueError("alignment-buffer goal artifact is malformed")
    criterion = str(goal.get("frontier_criterion", ""))
    if (
        not criterion
        or artifact.get("criterion_sha256")
        != hashlib.sha256(criterion.encode("utf-8")).hexdigest()
    ):
        raise ValueError("alignment-buffer goal artifact criterion binding differs")
    expected_training_goal = {**dict(goal), "text": criterion}
    if buffer.get("training_goal") != expected_training_goal:
        raise ValueError(
            "alignment buffer training goal differs from its immutable goal artifact"
        )
    return goal_id, expected_training_goal


def _reconstruct_cumulative_replay(
    *,
    previous: Mapping[str, Any] | None,
    goal_id: str,
    round_index: int,
    alignment_buffer_sha256: str,
    candidate_pool_sha256: str,
    positive_labels: set[str],
    negative_labels: set[str],
) -> dict[str, Any]:
    """Reconstruct one terminal cumulative-replay step, resetting on goal edits."""

    same_segment = previous is not None and previous.get("goal_id") == goal_id
    positive_raw = set(previous.get("positive_raw", ())) if same_segment else set()
    negative_raw = set(previous.get("negative_raw", ())) if same_segment else set()
    source_buffers = (
        [dict(row) for row in previous.get("source_buffers", ())]
        if same_segment
        else []
    )
    positive_raw.update(positive_labels)
    negative_raw.update(negative_labels)
    conflicts = positive_raw.intersection(negative_raw)
    positive_train = positive_raw.difference(conflicts)
    negative_train = negative_raw.difference(conflicts)
    source_buffers.append(
        {
            "round_index": round_index,
            "alignment_buffer_sha256": alignment_buffer_sha256,
            "candidate_pool_sha256": candidate_pool_sha256,
            "positive_count": len(positive_labels),
            "negative_count": len(negative_labels),
        }
    )
    alignment_policy = {
        "mode": "cumulative_within_current_goal",
        "fixed_goal_id": goal_id,
        "source_generator_ids": [
            f"p{row['round_index']}" for row in source_buffers
        ],
        "current_candidate_count": 100,
        "previous_round_labels_replayed": len(source_buffers) > 1,
        "reset_on_goal_transition": True,
        "stable_positive_and_negative_labels_used": True,
        "cumulative_positive_count": len(positive_train),
        "cumulative_negative_count": len(negative_train),
        "cross_round_conflicts_excluded": len(conflicts),
    }
    buffer_ablation = {
        "mode": "cumulative",
        "source_buffers": source_buffers,
        "cross_round_conflicts_excluded": sorted(conflicts),
        "positive_count": len(positive_train),
        "negative_count": len(negative_train),
    }
    return {
        "goal_id": goal_id,
        "positive_raw": sorted(positive_raw),
        "negative_raw": sorted(negative_raw),
        "positive_train": sorted(positive_train),
        "negative_train": sorted(negative_train),
        "source_buffers": source_buffers,
        "alignment_policy": alignment_policy,
        "buffer_ablation": buffer_ablation,
    }


def _validate_alignment_buffer(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    index: int,
    parent: Mapping[str, Any],
) -> tuple[dict[str, Any], set[str], set[str]]:
    round_root = output_root / "rounds" / f"round-{index:02d}"
    candidate_path = round_root / "candidates.jsonl"
    buffer_path = round_root / "alignment_buffer.json"
    buffer = read_json(buffer_path)
    candidate_rows = read_jsonl(candidate_path)
    parent_by_id = parent["by_id"]
    observed = {
        str(row.get("candidate_id")): str(row.get("sequence", "")).upper()
        for row in candidate_rows
    }
    flow_dpo = config.get("fine_tuning", {}).get("kind") == FLOW_DPO_KIND
    if (
        len(candidate_rows) != 100
        or observed != parent_by_id
        or buffer.get("candidate_pool_sha256") != sha256_file(candidate_path)
        or buffer.get("round_index") != index
        or buffer.get("generator_transition") != f"p{index}->p{index + 1}"
        or buffer.get("construction_candidate_ids") != list(parent_by_id)
        or buffer.get("fine_tuning_authorized") is not True
        or (
            buffer.get("negative_sequences_used_for_dpo_pairs") is not True
        )
        or buffer.get("llm_cardinal_scores_used") is not False
        or buffer.get("independent_generated_validation_cohort") is not False
        or buffer.get("working_cohort_reused_for_goal_selection_labels_and_training")
        is not True
        or float(buffer.get("negative_gradient_weight", -1.0))
        != (0.0 if flow_dpo else 0.3)
    ):
        raise ValueError(f"alignment buffer differs from current working cohort: {buffer_path}")
    _validate_alignment_goal_binding(output_root=output_root, buffer=buffer)
    source = buffer.get("candidate_source")
    if (
        not isinstance(source, Mapping)
        or source.get("kind") != "immutable_working_cohort"
        or source.get("source_generator_id") != f"p{index}"
        or source.get("role") != "working"
        or source.get("split_manifest_sha256")
        != sha256_file(parent["manifest_path"])
    ):
        raise ValueError("alignment-buffer distribution provenance differs")
    stability = buffer.get("repeat_label_stability")
    if not isinstance(stability, Mapping) or stability.get("sample_count") != 100:
        raise ValueError("alignment buffer omitted its stable categorical partition")
    expected_positive = {
        str(row["candidate_id"])
        for row in stability.get("outcomes", [])
        if row.get("stable_label") == "positive"
    }
    expected_negative = {
        str(row["candidate_id"])
        for row in stability.get("outcomes", [])
        if row.get("stable_label") == "negative"
    }
    positive = buffer.get("stable_positive")
    negative = buffer.get("stable_negative")
    if not isinstance(positive, list) or not isinstance(negative, list):
        raise ValueError("alignment buffer omitted stable all-label rows")
    positive_ids = {str(row.get("candidate_id")) for row in positive}
    negative_ids = {str(row.get("candidate_id")) for row in negative}
    if (
        positive_ids != expected_positive
        or negative_ids != expected_negative
        or len(positive_ids) < int(config["curriculum_discovery"]["minimum_stable_positive_count"])
        or len(negative_ids) < int(config["curriculum_discovery"]["minimum_stable_negative_count"])
        or any(parent_by_id.get(str(row.get("candidate_id"))) != row.get("sequence") for row in [*positive, *negative])
    ):
        raise ValueError("alignment buffer labels differ from stable outcomes")
    return (
        buffer,
        {str(row["sequence"]).upper() for row in positive},
        {str(row["sequence"]).upper() for row in negative},
    )


def _validate_updates(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    state: Mapping[str, Any],
    distributions: Mapping[str, Mapping[str, Any]],
    counts: Mapping[int, int],
) -> list[dict[str, Any]]:
    updates = state["updates"]
    if len(updates) != int(state["total_updates"]):
        raise ValueError("state update count differs from total_updates")
    update_paths = sorted(output_root.glob("updates/update-*.json"))
    if len(update_paths) != len(updates):
        raise ValueError("update ledger contains missing or unexpected receipts")
    p0_sequences = set(distributions["p0"]["by_id"].values())
    frozen_diversity = _average_pairwise_hamming(sorted(p0_sequences))
    frozen_composition = _amino_acid_distribution(sorted(p0_sequences))
    replay_sequences: set[str] = set()
    records: list[dict[str, Any]] = []
    previous_child_state: str | None = None
    replay_state: dict[str, Any] | None = None
    for index, state_update in enumerate(updates):
        if not isinstance(state_update, Mapping):
            raise ValueError("state update record is malformed")
        update_path = output_root / "updates" / f"update-{index:02d}.json"
        update = read_json(update_path)
        if update != state_update:
            raise ValueError("state and update receipt differ")
        parent_id, child_id = f"p{index}", f"p{index + 1}"
        flow_dpo = config.get("fine_tuning", {}).get("kind") == FLOW_DPO_KIND
        expected_negative_weight = 0.0 if flow_dpo else 0.3
        if (
            update.get("round_index") != index
            or update.get("parent_distribution_id") != parent_id
            or update.get("child_distribution_id") != child_id
            or update.get("checkpoint_gate_passed") is not True
        ):
            raise ValueError("update topology differs")
        buffer, positive_labels, negative_labels = _validate_alignment_buffer(
            config=config,
            output_root=output_root,
            index=index,
            parent=distributions[parent_id],
        )
        replay_sequences.update(positive_labels | negative_labels)
        summary_path = output_root / "rounds" / f"round-{index:02d}" / "training" / "summary.json"
        _verify_file(summary_path, update.get("training_summary_sha256"))
        if _path(update.get("training_summary_path", "")) != summary_path.resolve():
            raise ValueError("update training-summary path differs")
        summary = read_json(summary_path)
        if (
            summary.get("protocol_version") != PROTOCOL_VERSION
            or summary.get("experiment_id") != config.get("experiment_id")
            or summary.get("round_index") != index
            or summary.get("generator_transition") != f"{parent_id}->{child_id}"
            or summary.get("fresh_sampling_length_counts_per_arm")
            != {str(key): value for key, value in counts.items()}
            or summary.get("same_source_randomness_across_parent_and_child") is not False
        ):
            raise ValueError("GPU update-summary identity differs")
        validated = summary.get("validated_inputs")
        candidate_path = output_root / "rounds" / f"round-{index:02d}" / "candidates.jsonl"
        buffer_path = output_root / "rounds" / f"round-{index:02d}" / "alignment_buffer.json"
        if (
            not isinstance(validated, Mapping)
            or validated.get("config_sha256") != sha256_file(_path(state["config_path"]))
            or validated.get("alignment_buffer_sha256") != sha256_file(buffer_path)
            or validated.get("candidate_pool_sha256") != sha256_file(candidate_path)
            or validated.get("candidate_count") != 100
            or validated.get("generator_transition") != f"{parent_id}->{child_id}"
            or float(validated.get("negative_gradient_weight", -1.0))
            != expected_negative_weight
        ):
            raise ValueError("GPU validated-input receipt differs")
        if flow_dpo:
            positive_train = list(validated.get("positive_train", ()))
            negative_train = list(validated.get("negative_train", ()))
            pair_path = summary_path.parent / "preference_pairs.jsonl"
            pair_rows = read_jsonl(pair_path)
            expected_rows = []
            for positive_index, preferred in enumerate(sorted(set(positive_train))):
                for negative_index, rejected in enumerate(sorted(set(negative_train))):
                    expected_rows.append(
                        {
                            "pair_id": (
                                f"pair-p{positive_index:04d}-n{negative_index:04d}"
                            ),
                            "preferred_sequence": preferred,
                            "preferred_length": len(preferred),
                            "rejected_sequence": rejected,
                            "rejected_length": len(rejected),
                        }
                    )
            binding = validated.get("preference_pair_manifest")
            if (
                pair_rows != expected_rows
                or not isinstance(binding, Mapping)
                or _path(binding.get("path", "")) != pair_path.resolve()
                or binding.get("sha256") != sha256_file(pair_path)
                or binding.get("pair_count") != len(expected_rows)
                or binding.get("positive_count") != len(set(positive_train))
                or binding.get("negative_count") != len(set(negative_train))
                or binding.get("construction")
                != "full_cartesian_stable_positive_x_stable_negative"
            ):
                raise ValueError("flow-DPO Cartesian pair manifest differs")
            trace = summary.get("training_trace")
            trace_config = (
                trace.get("config") if isinstance(trace, Mapping) else None
            )
            checkpoint_states = summary.get("checkpoint_state_dict_sha256")
            expected_slots = int(config["fine_tuning"]["steps"]) * int(
                config["fine_tuning"]["batch_size"]
            )
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
            expected_trace_config = {
                name: config["fine_tuning"][name]
                for name in (
                    "learning_rate",
                    "steps",
                    "batch_size",
                    "beta",
                    "gradient_clip_norm",
                    "epsilon",
                    "seed",
                    "reference_policy",
                    "pair_construction",
                    "pair_schedule",
                    "corruption_coupling",
                    "loss_kind",
                )
            }
            expected_trace_config["seed"] = int(expected_trace_config["seed"]) + (
                index * int(config["curriculum_loop"]["round_seed_stride"])
            )
            if (
                not isinstance(trace, Mapping)
                or not isinstance(trace_config, Mapping)
                or not isinstance(checkpoint_states, Mapping)
                or trace.get("objective") != FLOW_DPO_KIND
                or trace.get("positive_count") != len(set(positive_train))
                or trace.get("negative_count") != len(set(negative_train))
                or trace.get("pair_count") != len(expected_rows)
                or trace.get("pair_manifest_path") != str(pair_path.resolve())
                or trace.get("pair_manifest_sha256") != sha256_file(pair_path)
                or trace.get("pair_slots_consumed") != expected_slots
                or trace.get("all_pairs_visited") is not True
                or int(trace.get("minimum_pair_visit_count", 0)) < 1
                or int(trace.get("maximum_pair_visit_count", 0))
                < int(trace.get("minimum_pair_visit_count", 0))
                or trace.get("complete_pair_epochs")
                != expected_slots // len(expected_rows)
                or trace.get("reference_checkpoint_unchanged") is not True
                or trace.get("checkpoint_sha256_before")
                != checkpoint_states.get("parent")
                or trace.get("checkpoint_sha256_after")
                != checkpoint_states.get("child")
                or trace.get("reference_checkpoint_sha256_before")
                != checkpoint_states.get("parent")
                or trace.get("reference_checkpoint_sha256_after")
                != checkpoint_states.get("parent")
                or trace.get("flow_dpo_surrogate_not_exact_sequence_log_likelihood")
                is not True
                or trace.get(
                    "training_batch_diagnostics_are_not_fixed_set_pre_post_metrics"
                )
                is not True
                or trace_config != expected_trace_config
                or any(name not in trace for name in required_batch_diagnostics)
            ):
                raise ValueError("flow-DPO training trace differs")

        replay = config.get("alignment_replay")
        if isinstance(replay, Mapping) and replay.get("mode") == (
            "cumulative_within_current_goal"
        ):
            goal_id = str(buffer.get("training_goal_id", ""))
            if not goal_id:
                raise ValueError("cumulative alignment buffer omitted its goal ID")
            replay_state = _reconstruct_cumulative_replay(
                previous=replay_state,
                goal_id=goal_id,
                round_index=index,
                alignment_buffer_sha256=sha256_file(buffer_path),
                candidate_pool_sha256=sha256_file(candidate_path),
                positive_labels=positive_labels,
                negative_labels=negative_labels,
            )
            if (
                list(validated.get("positive_train", ()))
                != replay_state["positive_train"]
                or list(validated.get("negative_train", ()))
                != replay_state["negative_train"]
                or validated.get("alignment_policy")
                != replay_state["alignment_policy"]
                or validated.get("buffer_ablation")
                != replay_state["buffer_ablation"]
            ):
                raise ValueError(
                    "cumulative within-goal training buffer was replaced or drifted"
                )

        checkpoint_hashes = summary.get("checkpoint_sha256")
        state_hashes = summary.get("checkpoint_state_dict_sha256")
        if not isinstance(checkpoint_hashes, Mapping) or not isinstance(state_hashes, Mapping):
            raise ValueError("GPU summary omitted checkpoint bindings")
        for arm in ("parent", "child"):
            checkpoint_path = summary_path.parent / "checkpoints" / f"{arm}.pt"
            _verify_file(checkpoint_path, checkpoint_hashes.get(arm))
            _require_sha256(state_hashes.get(arm), name=f"{arm} state_dict")
        if (
            config["checkpoint_gates"].get("require_checkpoint_state_change") is True
            and state_hashes["parent"] == state_hashes["child"]
        ):
            raise ValueError("checkpoint state-change gate was not met")
        if previous_child_state is not None and state_hashes["parent"] != previous_child_state:
            raise ValueError("checkpoint state_dict chain differs")
        previous_child_state = str(state_hashes["child"])
        child_checkpoint = summary.get("child_checkpoint_artifact")
        if (
            not isinstance(child_checkpoint, Mapping)
            or child_checkpoint.get("generator_id") != child_id
            or _path(child_checkpoint.get("path", ""))
            != (summary_path.parent / "checkpoints" / "child.pt").resolve()
            or child_checkpoint.get("file_sha256") != checkpoint_hashes["child"]
            or child_checkpoint.get("state_dict_sha256") != state_hashes["child"]
        ):
            raise ValueError("child checkpoint artifact differs")

        child_binding = summary.get("child_distribution_artifact")
        child_path = _verify_binding(child_binding, name="child distribution sample")
        if (
            child_binding.get("row_count") != 100
            or child_binding.get("generator_id") != child_id
            or child_path != distributions[child_id]["source_path"]
        ):
            raise ValueError("child distribution binding differs")
        child_rows = distributions[child_id]["rows"]
        child_sequences = [str(row["sequence"]).upper() for row in child_rows]
        child_unique = set(child_sequences)
        prior_sequences = set().union(
            *(set(distributions[f"p{prior}"]["by_id"].values()) for prior in range(index + 1))
        )
        replay_fraction = sum(seq in replay_sequences for seq in child_sequences) / 100
        novelty_fraction = sum(seq not in p0_sequences for seq in child_sequences) / 100
        overlap_fraction = len(child_unique.intersection(prior_sequences)) / len(child_unique)
        child_diversity = _average_pairwise_hamming(child_sequences)
        diversity_ratio = (
            None
            if child_diversity is None or frozen_diversity is None or frozen_diversity <= 0
            else child_diversity / frozen_diversity
        )
        jsd = _jensen_shannon(
            _amino_acid_distribution(child_sequences), frozen_composition
        )
        gates = config["checkpoint_gates"]
        expected_conditions = {
            "validity": 1.0 >= float(gates["minimum_valid_fraction"]),
            "unique_valid": len(child_unique) / 100 >= float(gates["minimum_unique_valid_fraction"]),
            "replay": replay_fraction <= float(gates["maximum_exact_replay_match_fraction"]),
            "novelty": novelty_fraction >= float(gates["minimum_novel_vs_original_pool_fraction"]),
            "diversity_vs_frozen_p0": diversity_ratio is not None and diversity_ratio >= float(gates["minimum_diversity_ratio_vs_frozen"]),
            "amino_acid_jsd_vs_frozen_p0": jsd <= float(gates["maximum_amino_acid_jsd_vs_frozen"]),
            "checkpoint_state_changed": state_hashes["parent"] != state_hashes["child"],
            "zero_overlap_with_prior_distributions": overlap_fraction <= float(gates["maximum_overlap_with_prior_distribution_fraction"]),
            "exact_100_candidate_cohort_eligibility": len(child_rows) == 100 and len(child_unique) == 100,
        }
        checkpoint_gate = summary.get("checkpoint_gate")
        if (
            not isinstance(checkpoint_gate, Mapping)
            or checkpoint_gate.get("conditions") != expected_conditions
            or checkpoint_gate.get("all_conditions_met") is not True
            or not all(expected_conditions.values())
        ):
            raise ValueError("checkpoint gates differ from CPU recomputation")
        child_summary = summary.get("sample_summaries", {}).get("child", {})
        expected_numeric = {
            "valid_count": 100.0,
            "valid_fraction": 1.0,
            "unique_valid_count": 100.0,
            "unique_valid_fraction_of_attempts": 1.0,
            "exact_replay_match_fraction": replay_fraction,
            "novel_vs_original_pool_fraction": novelty_fraction,
            "exact_overlap_with_prior_distribution_fraction": overlap_fraction,
            "amino_acid_jsd_vs_frozen_p0": jsd,
        }
        if any(not _same_number(child_summary.get(key), value) for key, value in expected_numeric.items()):
            raise ValueError("stored child non-collapse summary differs")
        if not _same_number(summary.get("diversity_ratio_child_vs_frozen_p0"), float(diversity_ratio)):
            raise ValueError("stored child diversity ratio differs")
        prior_record = summary.get("prior_distribution_overlap")
        if (
            not isinstance(prior_record, Mapping)
            or prior_record.get("count") != len(child_unique.intersection(prior_sequences))
            or not _same_number(prior_record.get("fraction_of_unique_valid_child"), overlap_fraction)
        ):
            raise ValueError("prior-distribution overlap receipt differs")
        for binding_name in ("private_predictor_scores", "frozen_evidence_reference"):
            binding = summary.get(binding_name)
            if not isinstance(binding, Mapping):
                raise ValueError(f"GPU summary omitted {binding_name}")
            _verify_binding(binding, name=binding_name)
        if summary["private_predictor_scores"].get(
            "cardinal_values_visible_to_llm"
        ) is not False:
            raise ValueError("private cardinal predictor values were exposed to the LLM")
        if summary["frozen_evidence_reference"].get("sha256") != config["fresh_sampling"]["frozen_evidence_reference"]["sha256"]:
            raise ValueError("GPU update used a different frozen evidence reference")
        for binding in summary.get("resumability", {}).values():
            _verify_binding(binding, name="GPU resumability receipt")
        child_manifest_path = distributions[child_id]["manifest_path"]
        if (
            _path(update.get("child_manifest_path", "")) != child_manifest_path.resolve()
            or update.get("child_manifest_sha256") != sha256_file(child_manifest_path)
            or update.get("trained_goal_id") != buffer.get("training_goal_id")
        ):
            raise ValueError("update receipt does not bind its child or training goal")
        records.append(
            {
                "round_index": index,
                "update_path": str(update_path.resolve()),
                "update_sha256": sha256_file(update_path),
                "summary_path": str(summary_path.resolve()),
                "summary_sha256": sha256_file(summary_path),
                "checkpoint_gate_recomputed": expected_conditions,
            }
        )
    return records


def _attempt_ledger(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    totals = {"attempts": 0, "prompt_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0}
    for role_name in ROLE_NAMES:
        role = config.get(role_name)
        if not isinstance(role, Mapping):
            raise ValueError(f"config omitted API role {role_name}")
        root = output_root / "api_attempts" / role_name
        attempts = prompt_tokens = output_tokens = 0
        for directory in sorted(root.iterdir()) if root.exists() else []:
            if not directory.is_dir() or SHA256_RE.fullmatch(directory.name) is None:
                raise ValueError("malformed scientific API-attempt directory")
            started = read_json(directory / "attempt_started.json")
            if started.get("request_hash") != directory.name or started.get("automatic_retries") != 0:
                raise ValueError("scientific API start receipt differs")
            attempts += 1
            response = read_json(directory / "provider_response.json")
            raw = response.get("provider_response")
            if (
                response.get("request_hash") != directory.name
                or not isinstance(response.get("http_status"), int)
                or not 200 <= response["http_status"] < 300
                or not isinstance(raw, Mapping)
            ):
                raise ValueError("terminal API response receipt is incomplete")
            marker = read_json(directory / "parsed_response.json")
            raw_hash = hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
            if (
                marker.get("request_hash") != directory.name
                or marker.get("provider_response_sha256") != raw_hash
                or marker.get("model_version") != str(raw.get("modelVersion", "")).strip()
            ):
                raise ValueError("terminal API parse marker differs")
            usage = raw.get("usageMetadata")
            if not isinstance(usage, Mapping):
                raise ValueError("terminal API response omitted usage")
            prompt = usage.get("promptTokenCount")
            total = usage.get("totalTokenCount")
            if (
                isinstance(prompt, bool)
                or not isinstance(prompt, int)
                or prompt < 0
                or isinstance(total, bool)
                or not isinstance(total, int)
                or total < prompt
            ):
                raise ValueError("terminal API usage is invalid")
            prompt_tokens += prompt
            output_tokens += total - prompt
        budget = role["api_budget"]
        cost = (
            prompt_tokens * float(budget["input_usd_per_million_tokens"])
            + output_tokens * float(budget["output_usd_per_million_tokens"])
        ) / 1_000_000
        within = (
            attempts <= int(role["maximum_provider_calls"])
            and prompt_tokens <= int(budget["maximum_prompt_tokens"])
            and output_tokens <= int(budget["maximum_output_tokens"])
            and cost <= float(budget["maximum_estimated_usd"])
        )
        if not within:
            raise PermissionError(f"{role_name} API ceiling was exceeded")
        roles[role_name] = {
            "attempts": attempts,
            "prompt_tokens": prompt_tokens,
            "output_tokens_including_reasoning": output_tokens,
            "estimated_usd": cost,
            "within_role_budget": True,
        }
        totals["attempts"] += attempts
        totals["prompt_tokens"] += prompt_tokens
        totals["output_tokens"] += output_tokens
        totals["estimated_usd"] += cost

    aggregate = config["aggregate_api_budget"]
    preflight_root = output_root / "preflight" / "allocations"
    allocation_paths = sorted(path for path in preflight_root.glob("*") if path.is_dir())
    preflight_attempts = preflight_prompt = preflight_output = 0
    successful_allocations = 0
    for allocation in allocation_paths:
        if not allocation.name.isdigit():
            raise ValueError("preflight allocation ID is malformed")
        summary_path = allocation / "summary.json"
        if not summary_path.is_file():
            raise ValueError("preflight allocation omitted its summary")
        allocation_summary = read_json(summary_path)
        entitlements = allocation_summary.get("generation_entitlement")
        if (
            allocation_summary.get("allocation_id") != allocation.name
            or not isinstance(entitlements, Mapping)
            or not entitlements
        ):
            raise ValueError("preflight allocation summary differs")
        allocation_success = False
        observed_models: set[str] = set()
        for directory in sorted((allocation / "attempts").glob("*")):
            if not directory.is_dir() or SHA256_RE.fullmatch(directory.name) is None:
                raise ValueError("preflight API-attempt directory is malformed")
            started = read_json(directory / "attempt_started.json")
            response = read_json(directory / "provider_response.json")
            raw = response.get("provider_response")
            marker = read_json(directory / "parsed_response.json")
            request_hash = str(started.get("request_hash", ""))
            requested_model = str(marker.get("requested_model", ""))
            request_body = started.get("provider_request_body")
            body_hash = (
                hashlib.sha256(canonical_json(request_body).encode("utf-8")).hexdigest()
                if isinstance(request_body, Mapping)
                else ""
            )
            entitlement = entitlements.get(requested_model)
            if (
                SHA256_RE.fullmatch(request_hash) is None
                or not requested_model
                or directory.name
                != hashlib.sha256(requested_model.encode("utf-8")).hexdigest()
                or requested_model in observed_models
                or started.get("provider_request_body_sha256") != body_hash
                or started.get("automatic_retries") != 0
                or response.get("request_hash") != request_hash
                or response.get("provider_request_body_sha256") != body_hash
                or not isinstance(response.get("http_status"), int)
                or not 200 <= response["http_status"] < 300
                or not isinstance(raw, Mapping)
                or marker.get("request_hash") != request_hash
                or marker.get("provider_response_sha256")
                != hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()
                or marker.get("model_version") != str(raw.get("modelVersion", "")).strip()
                or not isinstance(entitlement, Mapping)
                or entitlement.get("allocation_id") != allocation.name
                or entitlement.get("request_hash") != request_hash
                or entitlement.get("model") != requested_model
            ):
                raise ValueError("preflight API receipt differs")
            observed_models.add(requested_model)
            usage = raw.get("usageMetadata")
            if not isinstance(usage, Mapping):
                raise ValueError("preflight response omitted usage")
            prompt, total = usage.get("promptTokenCount"), usage.get("totalTokenCount")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (prompt, total)) or total < prompt:
                raise ValueError("preflight token usage differs")
            preflight_attempts += 1
            preflight_prompt += prompt
            preflight_output += total - prompt
            allocation_success = True
        if observed_models != set(str(key) for key in entitlements):
            raise ValueError("preflight allocation model coverage differs")
        if allocation_success:
            successful_allocations += 1
    if not allocation_paths or not successful_allocations:
        raise ValueError("terminal campaign lacks a successful preflight allocation")
    preflight_cost = (
        preflight_prompt * float(aggregate["preflight_input_usd_per_million_tokens"])
        + preflight_output * float(aggregate["preflight_output_usd_per_million_tokens"])
    ) / 1_000_000
    if (
        len(allocation_paths) > int(aggregate["maximum_preflight_allocations"])
        or preflight_attempts > int(aggregate["maximum_preflight_generation_calls"])
        or preflight_prompt > int(aggregate["maximum_preflight_prompt_tokens"])
        or preflight_output > int(aggregate["maximum_preflight_output_tokens"])
        or preflight_cost > float(aggregate["maximum_preflight_estimated_usd"])
    ):
        raise PermissionError("preflight API ceiling was exceeded")
    totals["attempts"] += preflight_attempts
    totals["prompt_tokens"] += preflight_prompt
    totals["output_tokens"] += preflight_output
    totals["estimated_usd"] += preflight_cost
    if (
        totals["attempts"] > int(aggregate["maximum_paid_calls"])
        or totals["prompt_tokens"] > int(aggregate["maximum_prompt_tokens"])
        or totals["output_tokens"] > int(aggregate["maximum_output_tokens"])
        or totals["estimated_usd"] > float(aggregate["maximum_estimated_usd"])
    ):
        raise PermissionError("aggregate API ceiling was exceeded")
    return {
        "roles": roles,
        "preflight": {
            "allocation_count": len(allocation_paths),
            "successful_allocation_count": successful_allocations,
            "attempts": preflight_attempts,
            "prompt_tokens": preflight_prompt,
            "output_tokens_including_reasoning": preflight_output,
            "estimated_usd": preflight_cost,
        },
        "totals": {**totals, "within_aggregate_budget": True},
    }


def _completed_transition_count(state: Mapping[str, Any]) -> int:
    return sum(
        str(row.get("kind")) in COUNTED_TRANSITIONS
        for row in state.get("accepted_transitions", [])
        if isinstance(row, Mapping)
    )


def _uses_exact_final_primary(config: Mapping[str, Any]) -> bool:
    return (
        config.get("evaluation", {}).get("primary_metric")
        == EXACT_FINAL_PRIMARY_METRIC
    )


def _exact_final_newly_reachable(state: Mapping[str, Any]) -> bool:
    return any(
        str(row.get("kind")) == "exact_final_newly_reachable"
        for row in state.get("accepted_transitions", [])
        if isinstance(row, Mapping)
    )


def _primary_metric_report(
    *, config: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    if _uses_exact_final_primary(config):
        reached = _exact_final_newly_reachable(state)
        return {
            "name": EXACT_FINAL_PRIMARY_METRIC,
            "value": reached,
            "success_value": True,
            "success": reached,
            "exact_final_trainable_at_p0_excluded": True,
        }
    count = _completed_transition_count(state)
    threshold = int(config["evaluation"]["success_transition_count"])
    return {
        "name": TRANSITION_COUNT_METRIC,
        "value": count,
        "success_threshold": threshold,
        "success": count >= threshold,
        "initial_broadening_counted": False,
        "maximum_semantic_tightenings": int(
            config["geometric_control"]["maximum_semantic_tightenings"]
        ),
    }


def _secondary_metric_reports(
    *, config: Mapping[str, Any], state: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not _uses_exact_final_primary(config):
        return []
    count = _completed_transition_count(state)
    reference = int(config["evaluation"]["secondary_reference_transition_count"])
    return [
        {
            "name": TRANSITION_COUNT_METRIC,
            "value": count,
            "reference_threshold": reference,
            "reference_threshold_met": count >= reference,
            "initial_broadening_counted": False,
            "maximum_semantic_tightenings": int(
                config["geometric_control"]["maximum_semantic_tightenings"]
            ),
            "role": "secondary_mechanistic_metric",
        }
    ]


def _campaign_path(value: Any, *, output_root: Path, name: str) -> Path:
    path = _path(value)
    try:
        path.relative_to(output_root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} points outside the campaign artifact root") from error
    return path


def _validate_goals(
    *, config: Mapping[str, Any], output_root: Path, state: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    goal_ids = [str(value) for value in state["goal_ids"]]
    intermediates = [goal_id for goal_id in goal_ids if goal_id != FINAL_GOAL_ID]
    if goal_ids[:1] != [FINAL_GOAL_ID] or intermediates != [
        f"G{index:02d}" for index in range(1, len(intermediates) + 1)
    ]:
        raise ValueError("accepted goal IDs do not form F,G01,G02,... topology")
    maximum_intermediate_goals = config.get("geometric_control", {}).get(
        "maximum_intermediate_goals"
    )
    if (
        maximum_intermediate_goals is not None
        and len(intermediates) > int(maximum_intermediate_goals)
    ):
        raise ValueError("state exceeds maximum intermediate-goal count")
    paths = {path.stem: path for path in (output_root / "goals").glob("*.json")}
    if set(paths) != set(goal_ids):
        raise ValueError("immutable goal artifacts differ from state goal_ids")
    goals: dict[str, dict[str, Any]] = {}
    for goal_id in goal_ids:
        value = read_json(paths[goal_id])
        goal = value.get("goal")
        if not isinstance(goal, Mapping):
            raise ValueError("goal artifact omitted its criterion")
        criterion = str(goal.get("frontier_criterion", ""))
        if (
            value.get("schema_version") != 1
            or value.get("goal_id") != goal_id
            or not criterion
            or value.get("criterion_sha256")
            != hashlib.sha256(criterion.encode("utf-8")).hexdigest()
            or goal.get("final_goal") != config["final_goal"]
            or (goal_id == FINAL_GOAL_ID) is not (goal.get("is_final") is True)
            or (
                goal_id == FINAL_GOAL_ID
                and criterion != config["final_goal"]
            )
            or (
                goal_id != FINAL_GOAL_ID
                and criterion == config["final_goal"]
            )
        ):
            raise ValueError(f"immutable goal artifact is malformed: {paths[goal_id]}")
        goals[goal_id] = value
    return goals


def _validate_agent_cache(
    *,
    output_root: Path,
    role: str,
    metadata: Any,
    expected_parsed: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{role} proposal/audit omitted provider metadata")
    request_hash = _require_sha256(
        metadata.get("request_hash"), name=f"{role} request hash"
    )
    root = output_root / "api_artifacts" / role / request_hash
    request = read_json(root / "request.json")
    response = read_json(root / "response.json")
    if (
        request.get("request_hash") != request_hash
        or hashlib.sha256(
            canonical_json(request.get("request")).encode("utf-8")
        ).hexdigest()
        != request_hash
        or response.get("request_hash") != request_hash
    ):
        raise ValueError(f"{role} cached request binding differs")
    response_content = {
        key: response.get(key)
        for key in ("response_id", "parsed", "usage", "raw_response")
    }
    if response.get("response_content_hash") != hashlib.sha256(
        canonical_json(response_content).encode("utf-8")
    ).hexdigest():
        raise ValueError(f"{role} cached response content hash differs")
    if response.get("parsed") != dict(expected_parsed):
        raise ValueError(f"{role} accepted record differs from cached parsed response")
    request_value = request.get("request")
    if not isinstance(request_value, Mapping):
        raise ValueError(f"{role} cached request is malformed")
    return request_value


def _cell_record(
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    distribution_id: str,
    goal_id: str,
    judgment_key: str,
) -> Mapping[str, Any]:
    key = (distribution_id, goal_id, judgment_key)
    if key not in cells:
        raise ValueError(
            "accepted transition lacks its immutable current/entry categorical cell: "
            f"{key}"
        )
    return cells[key]


def _validate_proposal_evidence(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    goals: Mapping[str, Mapping[str, Any]],
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    action: str,
    source_goal_id: str,
    target_goal_id: str,
    current_distribution_id: str,
    entry_distribution_id: str | None,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, dict[str, Any]]:
    target_artifact = goals.get(target_goal_id)
    if target_artifact is None:
        raise ValueError("accepted transition target lacks an immutable goal artifact")
    source = target_artifact.get("source")
    expected_kind = (
        "accepted_initial_broadening"
        if action == "broaden"
        else "accepted_newly_reachable_tightening"
    )
    if not isinstance(source, Mapping) or source.get("kind") != expected_kind:
        raise ValueError("accepted target goal lacks proposal provenance")
    proposal_path = _campaign_path(
        source.get("proposal_path", ""),
        output_root=output_root,
        name="accepted proposal",
    )
    decision_path = _campaign_path(
        source.get("decision_path", ""),
        output_root=output_root,
        name="accepted proposal decision",
    )
    _verify_file(proposal_path, source.get("proposal_sha256"))
    _verify_file(decision_path, source.get("decision_sha256"))
    proposal_artifact = read_json(proposal_path)
    decision = read_json(decision_path)
    proposal_name = str(proposal_artifact.get("proposal_name", ""))
    proposal = proposal_artifact.get("proposal")
    semantic = proposal_artifact.get("semantic_audit")
    if not isinstance(proposal, Mapping) or not isinstance(semantic, Mapping):
        raise ValueError("accepted proposal or semantic audit is malformed")
    expected_prefix = f"{current_distribution_id}-{action}-{target_goal_id}-attempt-"
    if (
        not proposal_name.startswith(expected_prefix)
        or proposal_path != (output_root / "proposals" / f"{proposal_name}.json").resolve()
        or decision_path != (output_root / "decisions" / f"{proposal_name}.json").resolve()
        or proposal.get("action") != action
        or proposal.get("goal") != target_artifact["goal"]
        or proposal_artifact.get("accepted") is not True
        or proposal_artifact.get("campaign_length_policy_repeated_exactly") is not True
        or proposal_artifact.get("proposed_witnesses_match_stable_categorical_outcomes") is not True
        or proposal_artifact.get("same_100_candidate_working_cohort_reused") is not True
        or proposal_artifact.get("cardinal_scores_used") is not False
        or decision.get("proposal_path") != str(proposal_path)
        or decision.get("proposal_sha256") != sha256_file(proposal_path)
        or decision.get("accepted") is not True
        or decision.get("correction_available") is not False
    ):
        raise ValueError("accepted proposal/decision evidence differs")
    allocations = proposal.get("length_policy", {}).get("allocations")
    if not isinstance(allocations, list) or not all(
        isinstance(row, Mapping) for row in allocations
    ) or {
        int(row["length"]): int(row["count"])
        for row in allocations
        if isinstance(row, Mapping)
    } != _length_counts(config):
        raise ValueError("accepted proposal changed the campaign length policy")

    required_semantic = {
        "decision": "accept",
        "requested_action_followed": True,
        "direction_toward_final_preserved": True,
        "directional_relation": (
            "source_subset_of_proposed"
            if action == "broaden"
            else "proposed_subset_of_source"
        ),
        "not_numeric_cutoff_or_synonym": True,
        "proxy_limitations_explicit": True,
        "multiple_routes_preserved_where_reasonable": True,
        "biological_success_not_claimed": True,
        "accepted": True,
    }
    if action == "broaden":
        required_semantic["materially_easier_if_broadening"] = True
    else:
        required_semantic["exactly_one_final_requirement_restored"] = True
    if any(semantic.get(key) != value for key, value in required_semantic.items()):
        raise ValueError("accepted proposal lacks a passing independent semantic audit")
    controller_request = _validate_agent_cache(
        output_root=output_root,
        role="curriculum_controller",
        metadata=proposal.get("metadata"),
        expected_parsed=dict(proposal.get("raw_response", {})),
    )
    controller_payload = controller_request.get("user_payload")
    controller_context = controller_request.get("logical_context")
    if (
        not isinstance(controller_payload, Mapping)
        or not isinstance(controller_context, Mapping)
        or controller_payload.get("requested_action_chosen_by_host") != action
        or controller_payload.get("immutable_final_goal") != config["final_goal"]
        or controller_payload.get("current_goal") != goals[source_goal_id]["goal"]
        or controller_payload.get("current_generator_id") != current_distribution_id
        or controller_payload.get("goal_entry_generator_id") != entry_distribution_id
        or controller_payload.get("working_cohort_size") != 100
        or controller_payload.get("no_independent_gate_cohort") is not True
        or controller_payload.get("cardinal_scores_forbidden") is not True
        or controller_context.get("action") != action
        or controller_context.get("generator_id") != current_distribution_id
        or controller_context.get("goal_entry_generator_id") != entry_distribution_id
        or controller_context.get("current_goal")
        != goals[source_goal_id]["goal"]["frontier_criterion"]
        or controller_context.get("proposal_attempt")
        != proposal.get("proposal_attempt")
    ):
        raise ValueError("accepted controller request has different semantic topology")
    semantic_metadata_keys = {"request_hash", "usage", "model_version", "thought_summary"}
    semantic_parsed = {
        key: value
        for key, value in semantic.items()
        if key not in semantic_metadata_keys and key != "accepted"
    }
    auditor_request = _validate_agent_cache(
        output_root=output_root,
        role="independent_auditor",
        metadata=semantic,
        expected_parsed=semantic_parsed,
    )
    auditor_payload = auditor_request.get("user_payload")
    expected_audited_proposal = {
        "action": action,
        "goal": target_artifact["goal"],
        "raw_response": proposal.get("raw_response"),
        "metadata": proposal.get("metadata"),
        "positive_witness_candidate_id": proposal.get(
            "positive_witness_candidate_id"
        ),
        "negative_witness_candidate_id": proposal.get(
            "negative_witness_candidate_id"
        ),
    }
    if (
        not isinstance(auditor_payload, Mapping)
        or auditor_payload.get("immutable_final_goal") != config["final_goal"]
        or auditor_payload.get("requested_action") != action
        or auditor_payload.get("source_goal") != goals[source_goal_id]["goal"]
        or auditor_payload.get("proposed_goal") != expected_audited_proposal
        or auditor_payload.get("gate_training_confirmation_outcomes_hidden")
        is not True
        or auditor_payload.get("cardinal_scores_forbidden") is not True
    ):
        raise ValueError("accepted independent audit has different semantic topology")

    current_record = _cell_record(
        cells,
        distribution_id=current_distribution_id,
        goal_id=target_goal_id,
        judgment_key=proposal_name,
    )
    current_cell = current_record["cell"]
    entry_record: Mapping[str, Any] | None = None
    if action == "tighten":
        if entry_distribution_id is None:
            raise ValueError("tightening omitted its goal-entry distribution")
        entry_record = _cell_record(
            cells,
            distribution_id=entry_distribution_id,
            goal_id=target_goal_id,
            judgment_key=f"{proposal_name}-goal-entry",
        )
    entry_cell = None if entry_record is None else entry_record["cell"]
    expected_learnability = json.loads(
        canonical_json(audit_learnability(current_cell).as_mapping())
    )
    if (
        proposal_artifact.get("current_categorical_cell")
        != current_cell.as_mapping()
        or proposal_artifact.get("goal_entry_categorical_cell")
        != (None if entry_cell is None else entry_cell.as_mapping())
        or proposal_artifact.get("learnability")
        != expected_learnability
        or not audit_learnability(current_cell).passed
    ):
        raise ValueError("accepted proposal categorical evidence differs")
    stable = current_cell.summary.by_candidate_id()
    positive_id = str(proposal.get("positive_witness_candidate_id", ""))
    negative_id = str(proposal.get("negative_witness_candidate_id", ""))
    if (
        positive_id not in stable
        or stable[positive_id].stable_label.value != "positive"
        or negative_id not in stable
        or stable[negative_id].stable_label.value != "negative"
    ):
        raise ValueError("accepted proposal witnesses differ from stable labels")
    return current_record, entry_record, proposal_artifact


def _goal_entry_cell(
    *,
    output_root: Path,
    goals: Mapping[str, Mapping[str, Any]],
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    goal_id: str,
) -> Mapping[str, Any]:
    if goal_id == FINAL_GOAL_ID:
        return _cell_record(
            cells,
            distribution_id="p0",
            goal_id=FINAL_GOAL_ID,
            judgment_key="initial-exact-final",
        )
    source = goals[goal_id].get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source goal omitted adoption provenance")
    proposal_path = _campaign_path(
        source.get("proposal_path", ""), output_root=output_root, name="source goal proposal"
    )
    proposal = read_json(proposal_path)
    cell = proposal.get("current_categorical_cell")
    if not isinstance(cell, Mapping):
        raise ValueError("source goal proposal omitted its entry cell")
    generator_id = str(cell.get("generator_id", ""))
    return _cell_record(
        cells,
        distribution_id=generator_id,
        goal_id=goal_id,
        judgment_key=str(proposal.get("proposal_name", "")),
    )


def _validate_source_goal_movement(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    goals: Mapping[str, Mapping[str, Any]],
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    source_goal_id: str,
    current_distribution_id: str,
) -> None:
    entry_record = _goal_entry_cell(
        output_root=output_root, goals=goals, cells=cells, goal_id=source_goal_id
    )
    current_record = _cell_record(
        cells,
        distribution_id=current_distribution_id,
        goal_id=source_goal_id,
        judgment_key=f"accepted-{source_goal_id}",
    )
    entry_cell = entry_record["cell"]
    current_cell = current_record["cell"]
    minimum_gain = float(config["geometric_control"]["minimum_enrichment"])
    minimum_rate = float(
        config["geometric_control"]["minimum_enriched_positive_fraction"]
    )
    if (
        current_cell.q + 1e-12 < minimum_rate
        or current_cell.q - entry_cell.q + 1e-12 < minimum_gain
    ):
        raise ValueError("source goal was not enriched before semantic tightening")
    decision_path = output_root / "decisions" / f"after-{current_distribution_id}-{source_goal_id}.json"
    decision = read_json(decision_path)
    movement = decision.get("movement")
    if (
        decision.get("distribution_id") != current_distribution_id
        or decision.get("goal_id") != source_goal_id
        or not isinstance(movement, Mapping)
        or movement.get("action") != "tighten"
        or not _same_number(movement.get("entry_positive_fraction"), entry_cell.q)
        or not _same_number(movement.get("current_positive_fraction"), current_cell.q)
        or not _same_number(movement.get("enrichment"), current_cell.q - entry_cell.q)
        or decision.get("no_independent_generated_validation_cohort") is not True
    ):
        raise ValueError("source-goal movement decision differs from categorical cells")


def _exact_final_unlock_evidence(
    *,
    config: Mapping[str, Any],
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
    source_goal_id: str,
    current_distribution_id: str,
    entry_distribution_id: str,
) -> tuple[Any, Any, bool]:
    """Recompute the mandatory exact-F current/entry check after enrichment."""

    current_cell = _cell_record(
        cells,
        distribution_id=current_distribution_id,
        goal_id=FINAL_GOAL_ID,
        judgment_key=f"transition-{source_goal_id}-to-final-current",
    )["cell"]
    entry_cell = _cell_record(
        cells,
        distribution_id=entry_distribution_id,
        goal_id=FINAL_GOAL_ID,
        judgment_key=f"transition-{source_goal_id}-to-final-entry",
    )["cell"]
    unlocked = audit_learnability(current_cell).passed and (
        current_cell.q - entry_cell.q + 1e-12
        >= float(config["geometric_control"]["minimum_new_goal_unlock"])
    )
    return current_cell, entry_cell, unlocked


def _validate_transition_evidence(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    state: Mapping[str, Any],
    goals: Mapping[str, Mapping[str, Any]],
    distributions: Mapping[str, Mapping[str, Any]],
    cells: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    transitions = state["accepted_transitions"]
    allowed = {
        "initial_broadening",
        "exact_final_trainable_at_p0",
        *COUNTED_TRANSITIONS,
    }
    active_goal = FINAL_GOAL_ID
    intermediate_targets: list[str] = []
    exact_final_evidenced = False
    exact_final_at_p0_evidenced = False
    exact_final_newly_reachable_evidenced = False
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping) or transition.get("kind") not in allowed:
            raise ValueError("state contains an unsupported semantic transition")
        kind = str(transition["kind"])
        if kind == "exact_final_trainable_at_p0":
            if (
                index != 0
                or index != len(transitions) - 1
                or transition.get("target_goal_id") != FINAL_GOAL_ID
                or transition.get("current_distribution_id") != "p0"
                or state.get("current_goal_id") != FINAL_GOAL_ID
                or state.get("current_distribution_id") != "p0"
                or int(state.get("total_updates", -1)) != 0
            ):
                raise ValueError("exact-final-at-p0 transition topology differs")
            initial_record = _goal_entry_cell(
                output_root=output_root, goals=goals, cells=cells, goal_id=FINAL_GOAL_ID
            )
            initial = initial_record["cell"]
            learnability = audit_learnability(initial)
            judgment_path = _campaign_path(
                transition.get("judgment_cell_path", ""),
                output_root=output_root,
                name="exact-final-at-p0 judgment cell",
            )
            if (
                judgment_path
                != (Path(initial_record["root"]) / "stable_summary.json").resolve()
                or not _same_number(transition.get("q_current"), initial.q)
                or transition.get("learnability")
                != json.loads(canonical_json(learnability.as_mapping()))
                or not learnability.passed
            ):
                raise ValueError("exact final was not trainable at p0")
            exact_final_evidenced = True
            exact_final_at_p0_evidenced = True
            continue
        source_goal_id = str(transition.get("source_goal_id", ""))
        target_goal_id = str(transition.get("target_goal_id", ""))
        if source_goal_id != active_goal:
            raise ValueError("semantic transitions do not form a contiguous goal chain")
        if kind == "initial_broadening":
            if index != 0 or source_goal_id != FINAL_GOAL_ID or target_goal_id != "G01" or transition.get("distribution_id") != "p0":
                raise ValueError("initial broadening topology differs")
            current_record, _entry, proposal = _validate_proposal_evidence(
                config=config,
                output_root=output_root,
                goals=goals,
                cells=cells,
                action="broaden",
                source_goal_id=source_goal_id,
                target_goal_id=target_goal_id,
                current_distribution_id="p0",
                entry_distribution_id=None,
            )
            final_cell = _goal_entry_cell(
                output_root=output_root, goals=goals, cells=cells, goal_id=FINAL_GOAL_ID
            )["cell"]
            current_cell = current_record["cell"]
            gain = current_cell.q - final_cell.q
            if (
                gain + 1e-12
                < float(config["geometric_control"]["minimum_initial_broadening_gain"])
                or proposal.get("initial_broadening_gain_passed") is not True
                or not _same_number(proposal.get("initial_broadening_gain_over_exact_final"), gain)
                or _path(transition.get("proposal_path", ""))
                != _path(goals[target_goal_id]["source"]["proposal_path"])
            ):
                raise ValueError("initial broadening evidence differs")
            active_goal = target_goal_id
            intermediate_targets.append(target_goal_id)
            continue

        entry_id = str(transition.get("goal_entry_distribution_id", ""))
        current_id = str(transition.get("current_distribution_id", ""))
        source_entry = _goal_entry_cell(
            output_root=output_root,
            goals=goals,
            cells=cells,
            goal_id=source_goal_id,
        )
        if source_entry["cell"].generator_id != entry_id:
            raise ValueError(
                "tightening goal-entry distribution differs from source-goal adoption"
            )
        if (
            entry_id not in distributions
            or current_id not in distributions
            or int(DIST_RE.fullmatch(current_id).group(1))
            <= int(DIST_RE.fullmatch(entry_id).group(1))
        ):
            raise ValueError("semantic tightening did not follow a generator update")
        _validate_source_goal_movement(
            config=config,
            output_root=output_root,
            goals=goals,
            cells=cells,
            source_goal_id=source_goal_id,
            current_distribution_id=current_id,
        )
        final_current_cell, final_entry_cell, final_unlocked = (
            _exact_final_unlock_evidence(
                config=config,
                cells=cells,
                source_goal_id=source_goal_id,
                current_distribution_id=current_id,
                entry_distribution_id=entry_id,
            )
        )
        if kind == "newly_reachable_smallest_tightening":
            if final_unlocked:
                raise ValueError(
                    "intermediate tightening was accepted although exact F was unlocked"
                )
            expected_target = f"G{len(intermediate_targets) + 1:02d}"
            if target_goal_id != expected_target or target_goal_id not in goals:
                raise ValueError("tighter intermediate-goal topology differs")
            current_record, entry_record, _proposal = _validate_proposal_evidence(
                config=config,
                output_root=output_root,
                goals=goals,
                cells=cells,
                action="tighten",
                source_goal_id=source_goal_id,
                target_goal_id=target_goal_id,
                current_distribution_id=current_id,
                entry_distribution_id=entry_id,
            )
            assert entry_record is not None
            current_cell, entry_cell = current_record["cell"], entry_record["cell"]
            if _path(transition.get("proposal_path", "")) != _path(
                goals[target_goal_id]["source"]["proposal_path"]
            ):
                raise ValueError("tightening transition proposal binding differs")
            active_goal = target_goal_id
            intermediate_targets.append(target_goal_id)
        else:
            if target_goal_id != FINAL_GOAL_ID or index != len(transitions) - 1:
                raise ValueError("exact-final transition must terminate the semantic chain")
            current_cell, entry_cell = final_current_cell, final_entry_cell
            if not final_unlocked:
                raise ValueError("exact final goal was not newly reachable when accepted")
            active_goal = FINAL_GOAL_ID
            exact_final_evidenced = True
            exact_final_newly_reachable_evidenced = True
        unlock = current_cell.q - entry_cell.q
        if (
            unlock + 1e-12
            < float(config["geometric_control"]["minimum_new_goal_unlock"])
            or not _same_number(transition.get("q_entry"), entry_cell.q)
            or not _same_number(transition.get("q_current"), current_cell.q)
        ):
            raise ValueError("accepted transition lacks the registered ten-point unlock")

    if intermediate_targets != [goal_id for goal_id in state["goal_ids"] if goal_id != FINAL_GOAL_ID]:
        raise ValueError("goal artifacts are not explained by accepted semantic transitions")
    terminal_failed_exact_final_checked = False
    terminal_goal_enriched = False
    state_goal_id = str(state.get("current_goal_id", ""))
    state_distribution_id = str(state.get("current_distribution_id", ""))
    if active_goal != FINAL_GOAL_ID and state_goal_id == active_goal:
        decision_path = (
            output_root
            / "decisions"
            / f"after-{state_distribution_id}-{state_goal_id}.json"
        )
        if decision_path.is_file():
            decision = read_json(decision_path)
            movement = decision.get("movement")
            if isinstance(movement, Mapping) and movement.get("action") == "tighten":
                terminal_goal_enriched = True
                _validate_source_goal_movement(
                    config=config,
                    output_root=output_root,
                    goals=goals,
                    cells=cells,
                    source_goal_id=state_goal_id,
                    current_distribution_id=state_distribution_id,
                )
                entry_id = str(
                    _goal_entry_cell(
                        output_root=output_root,
                        goals=goals,
                        cells=cells,
                        goal_id=state_goal_id,
                    )["cell"].generator_id
                )
                _current, _entry, unlocked = _exact_final_unlock_evidence(
                    config=config,
                    cells=cells,
                    source_goal_id=state_goal_id,
                    current_distribution_id=state_distribution_id,
                    entry_distribution_id=entry_id,
                )
                if unlocked:
                    raise ValueError(
                        "terminal path rejected exact F although it was newly reachable"
                    )
                terminal_failed_exact_final_checked = True

    maximum_intermediate_goals = config.get("geometric_control", {}).get(
        "maximum_intermediate_goals"
    )
    if (
        maximum_intermediate_goals is not None
        and len(intermediate_targets) == int(maximum_intermediate_goals)
        and active_goal != FINAL_GOAL_ID
        and terminal_goal_enriched
        and not terminal_failed_exact_final_checked
    ):
        raise ValueError("terminal G05 path lacks the required failed exact-F check")

    count = _completed_transition_count(state)
    status = str(state.get("terminal_status", ""))
    if _uses_exact_final_primary(config):
        if status == "reached_final_goal":
            if not exact_final_newly_reachable_evidenced:
                raise ValueError(
                    "reached_final_goal lacks newly reachable exact-final evidence"
                )
        elif status == "ineligible_final_goal_reachable_at_p0":
            if (
                not exact_final_at_p0_evidenced
                or exact_final_newly_reachable_evidenced
                or int(state.get("total_updates", -1)) != 0
            ):
                raise ValueError("p0-ineligible status lacks exact-final baseline evidence")
        elif status == "falsified":
            if exact_final_evidenced:
                raise ValueError("falsified status contradicts exact-final evidence")
        else:
            raise ValueError("unsupported exact-final primary terminal status")
    else:
        threshold = int(config["evaluation"]["success_transition_count"])
        if exact_final_evidenced and status != "reached_final_goal":
            raise ValueError("exact-final evidence requires reached_final_goal status")
        if status == "reached_final_goal":
            if not exact_final_evidenced:
                raise ValueError("reached_final_goal lacks an exact-final transition")
        elif status == "primary_metric_success":
            if count < threshold:
                raise ValueError("primary_metric_success lacks enough counted transitions")
        elif status == "falsified":
            gpu_failure = str(state.get("terminal_reason", "")).startswith(
                "GPU transition failed registered non-collapse checks"
            )
            if count >= threshold and not gpu_failure:
                raise ValueError("falsified status contradicts the primary transition count")
        else:
            raise ValueError("unsupported terminal scientific status")
    return {
        "active_goal_at_terminal": active_goal,
        "counted_transition_count": count,
        "intermediate_goal_targets": intermediate_targets,
        "exact_final_transition_evidenced": exact_final_evidenced,
        "terminal_status_semantics_passed": True,
    }


def _validate_terminal_report(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    output_root: Path,
    state: Mapping[str, Any],
    categorical_records: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = output_root / "final_report.json"
    report = read_json(report_path)
    goal_rows = [read_json(output_root / "goals" / f"{goal_id}.json") for goal_id in state["goal_ids"]]
    count = _completed_transition_count(state)
    primary = _primary_metric_report(config=config, state=state)
    secondary = _secondary_metric_reports(config=config, state=state)
    if count > int(config["geometric_control"]["maximum_semantic_tightenings"]):
        raise ValueError("state exceeds maximum semantic tightenings")
    expected_fields = {
        "experiment_id": config["experiment_id"],
        "protocol_version": PROTOCOL_VERSION,
        "status": state["terminal_status"],
        "reason": state["terminal_reason"],
        "immutable_final_goal": config["final_goal"],
        "goal_evolution": goal_rows,
        "accepted_transitions": state["accepted_transitions"],
        "primary_metric": primary,
        "generator_updates": state["updates"],
        "categorical_report_artifacts": list(categorical_records),
        "external_dataset_or_reference_context_visible": False,
        "working_cohort_size": 100,
        "one_working_cohort_per_generator_state": True,
        "independent_generated_validation_cohort": False,
        "cardinal_scores_used": False,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
    }
    if "claim_boundary" in config:
        expected_fields["claim_boundary"] = str(config["claim_boundary"])
    if secondary:
        expected_fields["secondary_metrics"] = secondary
    if any(report.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("terminal report differs from latest immutable state")
    return report, primary


def _assert_no_cardinal_independent_or_external_reference(
    *, config: Mapping[str, Any], output_root: Path, state: Mapping[str, Any]
) -> None:
    artifacts: list[Path] = [
        output_root / "final_report.json",
        *[path for path, _ in _state_snapshots(output_root)],
        *sorted(output_root.glob("rounds/round-*/alignment_buffer.json")),
        *sorted(output_root.glob("judgments/*/*/categorical_report.json")),
        *sorted(output_root.glob("distributions/*/distribution_split_manifest.json")),
    ]
    for path in artifacts:
        value = read_json(path)
        for key in ("cardinal_scores_used", "llm_cardinal_scores_used"):
            if key in value and value[key] is not False:
                raise ValueError(f"cardinal LLM scores appear in {path}")
        if (
            "independent_generated_validation_cohort" in value
            and value["independent_generated_validation_cohort"] is not False
        ):
            raise ValueError(f"independent generated cohort appears in {path}")
        if (
            "external_dataset_or_reference_context_visible" in value
            and value["external_dataset_or_reference_context_visible"] is not False
        ):
            raise ValueError(f"external peptide reference appears in {path}")
    unexpected = [
        path
        for path in output_root.rglob("distribution_split_manifest.json")
        if path.parent.name not in set(state["distribution_ids"])
    ]
    if unexpected:
        raise ValueError("unregistered independent generated cohort exists")
    if (
        config["geometric_control"].get(
            "same_working_cohort_used_for_goal_selection_and_training"
        )
        is not True
        or config["evaluation"].get("same_cohort_selection_and_training") is not True
    ):
        raise ValueError("same-cohort V13 contract differs")
    forbidden_artifacts = [
        *output_root.rglob("*reference_novelty*"),
        *output_root.rglob("*anticancer_reference*"),
        *output_root.rglob("*private_reference*"),
    ]
    if forbidden_artifacts:
        raise ValueError("external peptide reference artifact exists in V13 output")
    forbidden_request_keys = (
        '"compact_anticancer_reference_context"',
        '"anticancer_reference_context"',
        '"reference_peptides"',
        '"labelled_examples"',
    )
    request_paths = [
        *output_root.glob("api_attempts/*/*/request.json"),
        *output_root.glob("api_artifacts/*/*/request.json"),
    ]
    for path in sorted(request_paths):
        text = path.read_text(encoding="utf-8")
        if any(key in text for key in forbidden_request_keys):
            raise ValueError(f"external peptide reference leaked into {path}")
    forbidden_dataset_names = ("anticp2", "hemopi2", "dramp")
    for path in sorted(output_root.rglob("attempt_started.json")):
        started = read_json(path)
        body = started.get("provider_request_body")
        if body is None:
            continue
        text = canonical_json(body).casefold()
        if any(key in text for key in forbidden_request_keys) or any(
            name in text for name in forbidden_dataset_names
        ):
            raise ValueError(f"external peptide reference leaked into {path}")


def validate_terminal(
    *, config_path: Path, output_root: Path
) -> dict[str, Any]:
    """Validate one completed V13 artifact chain and publish its receipt."""

    config = read_json(config_path)
    counts, frozen_manifest = _validate_frozen_config(config_path, config)
    snapshots = _state_snapshots(output_root)
    latest_path, state = snapshots[-1]
    if (
        state.get("experiment_id") != config.get("experiment_id")
        or _path(state.get("config_path", "")) != config_path.resolve()
        or state.get("config_sha256") != sha256_file(config_path)
        or state.get("phase") != "terminal_complete"
        or not str(state.get("terminal_status", "")).strip()
        or state.get("cardinal_scores_used") is not False
        or state.get("independent_generated_validation_cohort") is not False
    ):
        raise ValueError("latest immutable state is not a bound terminal V13 state")
    distributions = _validate_distributions(
        config=config,
        output_root=output_root,
        state=state,
        counts=counts,
    )
    categorical, categorical_cells = _validate_categorical_reports(
        config=config, output_root=output_root, distributions=distributions
    )
    goals = _validate_goals(config=config, output_root=output_root, state=state)
    transition_evidence = _validate_transition_evidence(
        config=config,
        output_root=output_root,
        state=state,
        goals=goals,
        distributions=distributions,
        cells=categorical_cells,
    )
    updates = _validate_updates(
        config=config,
        output_root=output_root,
        state=state,
        distributions=distributions,
        counts=counts,
    )
    ledger = _attempt_ledger(config, output_root)
    report, primary = _validate_terminal_report(
        config_path=config_path,
        config=config,
        output_root=output_root,
        state=state,
        categorical_records=categorical,
    )
    _assert_no_cardinal_independent_or_external_reference(
        config=config, output_root=output_root, state=state
    )
    result = {
        "schema_version": 1,
        "status": "validated_terminal_geometric_v13_chain",
        "experiment_id": config["experiment_id"],
        "protocol_version": PROTOCOL_VERSION,
        "frozen_config_path": str(config_path.resolve()),
        "frozen_config_sha256": sha256_file(config_path),
        "frozen_config_manifest_sha256": sha256_file(
            config_path.with_name("frozen_config_manifest.json")
        ),
        "latest_state_snapshot_path": str(latest_path.resolve()),
        "latest_state_snapshot_sha256": sha256_file(latest_path),
        "snapshot_count": len(snapshots),
        "final_report_path": str((output_root / "final_report.json").resolve()),
        "final_report_sha256": sha256_file(output_root / "final_report.json"),
        "primary_metric_recomputed": primary,
        "distribution_count": len(distributions),
        "distribution_manifest_sha256": {
            distribution_id: sha256_file(value["manifest_path"])
            for distribution_id, value in distributions.items()
        },
        "completed_update_count": len(updates),
        "validated_updates": updates,
        "categorical_report_count": len(categorical),
        "categorical_report_artifacts": categorical,
        "external_peptide_dataset_absent": True,
        "api_ledger": ledger,
        "length_counts": {str(key): value for key, value in counts.items()},
        "initial_broadening_excluded_from_primary_metric": True,
        "transition_evidence_recomputed": transition_evidence,
        "cardinal_llm_scores_absent": True,
        "independent_generated_validation_cohort_absent": True,
        "frozen_manifest_transformation": frozen_manifest["transformation"],
        "claim_boundary": report.get("claim_boundary"),
    }
    immutable_json(output_root / "final_validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = validate_terminal(
        config_path=args.config.resolve(), output_root=args.output_root.resolve()
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
