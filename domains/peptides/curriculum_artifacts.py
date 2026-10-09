"""Frozen cohort splits and confirmation matrices for semantic staircases.

This module is deliberately independent of the LLM controller and movement
gates.  It has two small responsibilities:

* deterministically partition one balanced, 700-candidate generator sample
  into discovery, gate, train, confirmation, and reserve cohorts; and
* turn repeated categorical judgments on frozen confirmation cohorts into a
  content-addressed goal-by-generator matrix.

No cardinal LLM value is consumed or emitted.  All selections are hash based,
so they do not depend on process RNG state or input row order.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
SPLIT_PROTOCOL = "balanced-peptide-distribution-split-v1"
MATRIX_PROTOCOL = "repeated-categorical-goal-distribution-matrix-v1"
CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
REGISTERED_LENGTHS = (8, 10, 12, 16)
ROLE_SIZES: tuple[tuple[str, int], ...] = (
    ("discovery", 100),
    ("gate", 200),
    ("train", 100),
    ("confirmation", 200),
    ("reserve", 100),
)
ALLOWED_LABELS = frozenset({"positive", "negative", "abstain"})


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically while rejecting NaN and infinities."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hexadecimal digest") from error


def _content_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _immutable_write(path: Path, content: bytes) -> None:
    """Create an artifact atomically, permitting only byte-identical replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise FileExistsError(f"immutable artifact differs: {path}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    content = (
        json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _immutable_write(path, content)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )


def _normalized_candidates(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    sequences: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"candidate row {index} must be an object")
        candidate_id = row.get("candidate_id")
        sequence = row.get("sequence")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id != candidate_id.strip()
        ):
            raise ValueError(f"candidate row {index} has an invalid candidate_id")
        if (
            not isinstance(sequence, str)
            or not sequence
            or sequence != sequence.strip().upper()
            or set(sequence).difference(CANONICAL_AMINO_ACIDS)
        ):
            raise ValueError(f"candidate {candidate_id} has an invalid sequence")
        if len(sequence) not in REGISTERED_LENGTHS:
            raise ValueError(
                f"candidate {candidate_id} has unregistered length {len(sequence)}"
            )
        if candidate_id in ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        if sequence in sequences:
            raise ValueError(f"duplicate peptide sequence: {sequence}")
        ids.add(candidate_id)
        sequences.add(sequence)
        normalized.append(
            {
                "candidate_id": candidate_id,
                "sequence": sequence,
                "sequence_length": len(sequence),
            }
        )
    return tuple(sorted(normalized, key=lambda row: row["candidate_id"]))


def read_candidate_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"candidate JSONL row {line_number} is not an object")
            rows.append(value)
    return _normalized_candidates(rows)


def _selection_key(
    *,
    distribution_id: str,
    split_seed: int,
    candidate: Mapping[str, Any],
) -> str:
    material = {
        "protocol": SPLIT_PROTOCOL,
        "distribution_id": distribution_id,
        "split_seed": split_seed,
        "candidate_id": candidate["candidate_id"],
        "sequence": candidate["sequence"],
    }
    return _content_sha256(material)


def _role_order_key(
    *,
    distribution_id: str,
    split_seed: int,
    role: str,
    candidate: Mapping[str, Any],
) -> str:
    return _content_sha256(
        {
            "protocol": SPLIT_PROTOCOL,
            "purpose": "within-role-order",
            "distribution_id": distribution_id,
            "split_seed": split_seed,
            "role": role,
            "candidate_id": candidate["candidate_id"],
            "sequence": candidate["sequence"],
        }
    )


def build_balanced_distribution_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    distribution_id: str,
    split_seed: int,
    source_artifact_path: str,
    source_artifact_sha256: str,
    generator_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the registered 700-way split without touching process RNG state.

    The input must contain exactly 175 unique canonical sequences at each of
    lengths 8, 10, 12, and 16.  Each role receives an equal number from every
    length: 25/50/25/50/25 respectively.
    """

    if (
        not isinstance(distribution_id, str)
        or not distribution_id
        or distribution_id != distribution_id.strip()
    ):
        raise ValueError("distribution_id must be a non-empty trimmed string")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise TypeError("split_seed must be an integer")
    if not isinstance(source_artifact_path, str) or not source_artifact_path:
        raise ValueError("source_artifact_path must be non-empty")
    _validate_sha256(source_artifact_sha256, name="source_artifact_sha256")
    if not isinstance(generator_provenance, Mapping) or not generator_provenance:
        raise ValueError("generator_provenance must be a non-empty object")
    # Also rejects non-JSON types and non-finite numbers before selection.
    canonical_json(generator_provenance)

    candidates = _normalized_candidates(rows)
    expected_total = sum(size for _, size in ROLE_SIZES)
    if len(candidates) != expected_total:
        raise ValueError(f"expected exactly {expected_total} unique candidates")
    by_length: dict[int, list[dict[str, Any]]] = {
        length: [] for length in REGISTERED_LENGTHS
    }
    for candidate in candidates:
        by_length[candidate["sequence_length"]].append(candidate)
    expected_per_length = expected_total // len(REGISTERED_LENGTHS)
    length_counts = {length: len(by_length[length]) for length in REGISTERED_LENGTHS}
    if any(count != expected_per_length for count in length_counts.values()):
        raise ValueError(
            "candidate pool must contain exactly "
            f"{expected_per_length} peptides at each registered length"
        )

    for length in REGISTERED_LENGTHS:
        by_length[length].sort(
            key=lambda candidate: _selection_key(
                distribution_id=distribution_id,
                split_seed=split_seed,
                candidate=candidate,
            )
        )

    role_rows: dict[str, list[dict[str, Any]]] = {
        role: [] for role, _ in ROLE_SIZES
    }
    offsets = {length: 0 for length in REGISTERED_LENGTHS}
    role_artifacts: dict[str, dict[str, Any]] = {}
    for role, size in ROLE_SIZES:
        if size % len(REGISTERED_LENGTHS):
            raise AssertionError("registered role size is not length-balanced")
        quota = size // len(REGISTERED_LENGTHS)
        for length in REGISTERED_LENGTHS:
            start = offsets[length]
            stop = start + quota
            role_rows[role].extend(by_length[length][start:stop])
            offsets[length] = stop
        role_rows[role].sort(
            key=lambda candidate: _role_order_key(
                distribution_id=distribution_id,
                split_seed=split_seed,
                role=role,
                candidate=candidate,
            )
        )
        content = _jsonl_bytes(role_rows[role])
        role_artifacts[role] = {
            "relative_path": f"roles/{role}.jsonl",
            "sha256": sha256_bytes(content),
            "candidate_count": len(role_rows[role]),
            "length_counts": {
                str(length): sum(
                    row["sequence_length"] == length for row in role_rows[role]
                )
                for length in REGISTERED_LENGTHS
            },
            "candidates": role_rows[role],
        }
    if any(offset != expected_per_length for offset in offsets.values()):
        raise AssertionError("registered split did not consume the complete pool")

    candidate_projection = list(candidates)
    generator_value = json.loads(canonical_json(generator_provenance))
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": SPLIT_PROTOCOL,
        "distribution_id": distribution_id,
        "split_seed": split_seed,
        "registered_lengths": list(REGISTERED_LENGTHS),
        "candidate_count": len(candidates),
        "source_artifact": {
            "path": source_artifact_path,
            "sha256": source_artifact_sha256,
        },
        "candidate_projection_sha256": _content_sha256(candidate_projection),
        "generator_provenance": generator_value,
        "generator_provenance_sha256": _content_sha256(generator_value),
        "length_counts": {
            str(length): length_counts[length] for length in REGISTERED_LENGTHS
        },
        "role_order": [role for role, _ in ROLE_SIZES],
        "roles": role_artifacts,
        "selection": {
            "kind": "sha256_order_within_length_then_fixed_quota",
            "input_order_invariant": True,
            "process_rng_used": False,
            "identity_and_sequence_disjoint_across_roles": True,
        },
    }
    manifest["manifest_content_sha256"] = _content_sha256(manifest)
    validate_distribution_split_manifest(manifest)
    return manifest


def build_balanced_distribution_split_from_jsonl(
    candidate_path: Path,
    *,
    distribution_id: str,
    split_seed: int,
    generator_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Load, bind, and split one exact candidate JSONL artifact."""

    resolved = candidate_path.resolve()
    rows = read_candidate_jsonl(resolved)
    return build_balanced_distribution_split(
        rows,
        distribution_id=distribution_id,
        split_seed=split_seed,
        source_artifact_path=str(resolved),
        source_artifact_sha256=sha256_file(resolved),
        generator_provenance=generator_provenance,
    )


def validate_distribution_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate hashes, quotas, and global identity/sequence disjointness."""

    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported distribution-split schema")
    if manifest.get("protocol") != SPLIT_PROTOCOL:
        raise ValueError("unsupported distribution-split protocol")
    expected_hash = manifest.get("manifest_content_sha256")
    _validate_sha256(expected_hash, name="manifest_content_sha256")
    content = dict(manifest)
    del content["manifest_content_sha256"]
    if _content_sha256(content) != expected_hash:
        raise ValueError("distribution-split manifest content hash differs")
    source = manifest.get("source_artifact")
    if not isinstance(source, Mapping):
        raise ValueError("source_artifact must be an object")
    if not isinstance(source.get("path"), str) or not source["path"]:
        raise ValueError("source_artifact.path must be non-empty")
    _validate_sha256(source.get("sha256"), name="source_artifact.sha256")
    generator = manifest.get("generator_provenance")
    if not isinstance(generator, Mapping) or not generator:
        raise ValueError("generator_provenance must be a non-empty object")
    if _content_sha256(generator) != manifest.get("generator_provenance_sha256"):
        raise ValueError("generator provenance hash differs")
    if manifest.get("registered_lengths") != list(REGISTERED_LENGTHS):
        raise ValueError("registered peptide lengths differ")
    if manifest.get("role_order") != [role for role, _ in ROLE_SIZES]:
        raise ValueError("registered split role order differs")

    roles = manifest.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        role for role, _ in ROLE_SIZES
    }:
        raise ValueError("split manifest has the wrong role set")
    all_candidates: list[Mapping[str, Any]] = []
    all_ids: set[str] = set()
    all_sequences: set[str] = set()
    for role, expected_size in ROLE_SIZES:
        value = roles[role]
        if not isinstance(value, Mapping):
            raise ValueError(f"role {role} must be an object")
        if value.get("relative_path") != f"roles/{role}.jsonl":
            raise ValueError(f"role {role} artifact path differs")
        candidates = value.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"role {role} candidates must be a list")
        normalized = _normalized_candidates(candidates)
        # Preserve the registered within-role order when checking file hashes.
        if [row["candidate_id"] for row in candidates] != [
            row["candidate_id"] for row in normalized
        ]:
            # _normalized_candidates sorts by ID, whereas hash ordering normally
            # does not.  Re-normalize each row without changing input order.
            normalized_by_id = {row["candidate_id"]: row for row in normalized}
            ordered = [normalized_by_id[row["candidate_id"]] for row in candidates]
        else:
            ordered = list(normalized)
        if (
            len(ordered) != expected_size
            or value.get("candidate_count") != expected_size
        ):
            raise ValueError(f"role {role} has the wrong candidate count")
        expected_quota = expected_size // len(REGISTERED_LENGTHS)
        observed_lengths = Counter(row["sequence_length"] for row in ordered)
        expected_lengths = {
            str(length): expected_quota for length in REGISTERED_LENGTHS
        }
        if value.get("length_counts") != expected_lengths or any(
            observed_lengths[length] != expected_quota
            for length in REGISTERED_LENGTHS
        ):
            raise ValueError(f"role {role} is not balanced across peptide lengths")
        if sha256_bytes(_jsonl_bytes(ordered)) != value.get("sha256"):
            raise ValueError(f"role {role} JSONL hash differs")
        ids = {row["candidate_id"] for row in ordered}
        sequences = {row["sequence"] for row in ordered}
        if all_ids.intersection(ids) or all_sequences.intersection(sequences):
            raise ValueError("candidate IDs or sequences overlap across split roles")
        all_ids.update(ids)
        all_sequences.update(sequences)
        all_candidates.extend(ordered)
    if len(all_candidates) != sum(size for _, size in ROLE_SIZES):
        raise ValueError("split roles do not cover the registered candidate count")
    projection = sorted(all_candidates, key=lambda row: row["candidate_id"])
    if _content_sha256(projection) != manifest.get("candidate_projection_sha256"):
        raise ValueError("candidate projection hash differs")


def write_distribution_split(
    output_root: Path, manifest: Mapping[str, Any]
) -> Path:
    """Write role JSONLs followed by their immutable manifest."""

    validate_distribution_split_manifest(manifest)
    for role, _ in ROLE_SIZES:
        artifact = manifest["roles"][role]
        path = output_root / artifact["relative_path"]
        content = _jsonl_bytes(artifact["candidates"])
        if sha256_bytes(content) != artifact["sha256"]:
            raise ValueError(f"role {role} content changed before write")
        _immutable_write(path, content)
    manifest_path = output_root / "distribution_split_manifest.json"
    write_immutable_json(manifest_path, manifest)
    return manifest_path


@dataclass(frozen=True)
class FrozenGoalBinding:
    """One exact semantic goal and the artifact from which it originated."""

    goal_id: str
    criterion: str
    source_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.goal_id or self.goal_id != self.goal_id.strip():
            raise ValueError("goal_id must be a non-empty trimmed string")
        if not self.criterion or self.criterion != self.criterion.strip():
            raise ValueError("criterion must be a non-empty trimmed string")
        _validate_sha256(
            self.source_artifact_sha256, name="goal source_artifact_sha256"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "criterion": self.criterion,
            "criterion_sha256": sha256_bytes(self.criterion.encode("utf-8")),
            "source_artifact_sha256": self.source_artifact_sha256,
        }


@dataclass(frozen=True)
class ConfirmationDistributionBinding:
    """Confirmation cohort binding for one frozen generator distribution."""

    distribution_id: str
    split_manifest_sha256: str
    confirmation_role_sha256: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.distribution_id
            or self.distribution_id != self.distribution_id.strip()
        ):
            raise ValueError("distribution_id must be a non-empty trimmed string")
        _validate_sha256(self.split_manifest_sha256, name="split_manifest_sha256")
        _validate_sha256(
            self.confirmation_role_sha256, name="confirmation_role_sha256"
        )
        if len(self.candidate_ids) != 200 or len(self.candidate_ids) != len(
            set(self.candidate_ids)
        ):
            raise ValueError(
                "confirmation cohort must contain exactly 200 unique candidate IDs"
            )
        if any(not value or value != value.strip() for value in self.candidate_ids):
            raise ValueError("confirmation candidate IDs must be non-empty and trimmed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "distribution_id": self.distribution_id,
            "split_manifest_sha256": self.split_manifest_sha256,
            "confirmation_role_sha256": self.confirmation_role_sha256,
            "candidate_count": len(self.candidate_ids),
            "candidate_ids_sha256": _content_sha256(sorted(self.candidate_ids)),
        }


def confirmation_binding_from_split_manifest(
    manifest_path: Path,
) -> ConfirmationDistributionBinding:
    """Load the exact confirmation cohort from a written split manifest."""

    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("distribution split manifest must be a JSON object")
    validate_distribution_split_manifest(value)
    confirmation = value["roles"]["confirmation"]
    return ConfirmationDistributionBinding(
        distribution_id=value["distribution_id"],
        split_manifest_sha256=sha256_file(manifest_path),
        confirmation_role_sha256=confirmation["sha256"],
        candidate_ids=tuple(
            row["candidate_id"] for row in confirmation["candidates"]
        ),
    )


@dataclass(frozen=True)
class RepeatedJudgmentCell:
    """Two categorical passes for one frozen goal/distribution pair."""

    distribution_id: str
    goal_id: str
    first_labels: tuple[tuple[str, str], ...]
    repeat_labels: tuple[tuple[str, str], ...]
    first_judgment_artifact_sha256: str
    repeat_judgment_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.distribution_id or not self.goal_id:
            raise ValueError("matrix cell IDs must be non-empty")
        _validate_sha256(
            self.first_judgment_artifact_sha256,
            name="first_judgment_artifact_sha256",
        )
        _validate_sha256(
            self.repeat_judgment_artifact_sha256,
            name="repeat_judgment_artifact_sha256",
        )
        for name, values in (
            ("first_labels", self.first_labels),
            ("repeat_labels", self.repeat_labels),
        ):
            ids = [candidate_id for candidate_id, _ in values]
            if not ids or len(ids) != len(set(ids)):
                raise ValueError(f"{name} candidate IDs must be non-empty and unique")
            if any(not value or value != value.strip() for value in ids):
                raise ValueError(f"{name} candidate IDs must be non-empty and trimmed")
            if any(label not in ALLOWED_LABELS for _, label in values):
                raise ValueError(f"{name} contains an unsupported categorical label")


def _label_counts(labels: Mapping[str, str]) -> dict[str, int]:
    return {
        label: sum(value == label for value in labels.values())
        for label in ("positive", "negative", "abstain")
    }


def build_goal_distribution_matrix(
    *,
    goals: Sequence[FrozenGoalBinding],
    distributions: Sequence[ConfirmationDistributionBinding],
    cells: Sequence[RepeatedJudgmentCell],
    required_cell_keys: Sequence[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a repeated-label confirmation matrix.

    This function only summarizes caller-supplied frozen artifacts.  It cannot
    propose goals, call a judge, train a generator, or influence a gate.

    By default every goal/distribution cell is required, preserving the
    original complete-matrix protocol.  A registered experiment may instead
    supply an exact sparse set of cells.  Sparse coverage is explicit in the
    resulting artifact and cannot silently omit any caller-registered key.
    """

    if not goals or not distributions:
        raise ValueError("matrix requires at least one goal and distribution")
    goal_by_id = {goal.goal_id: goal for goal in goals}
    distribution_by_id = {
        distribution.distribution_id: distribution for distribution in distributions
    }
    if len(goal_by_id) != len(goals):
        raise ValueError("matrix contains duplicate goal IDs")
    if len(distribution_by_id) != len(distributions):
        raise ValueError("matrix contains duplicate distribution IDs")
    cell_by_key = {(cell.distribution_id, cell.goal_id): cell for cell in cells}
    if len(cell_by_key) != len(cells):
        raise ValueError("matrix contains duplicate goal/distribution cells")
    cartesian_keys = {
        (distribution_id, goal_id)
        for distribution_id in distribution_by_id
        for goal_id in goal_by_id
    }
    if required_cell_keys is None:
        expected_keys = cartesian_keys
        coverage = "complete_cartesian"
    else:
        expected_keys = set(required_cell_keys)
        if not expected_keys:
            raise ValueError("sparse matrix requires at least one registered cell")
        if len(expected_keys) != len(required_cell_keys):
            raise ValueError("sparse matrix contains duplicate registered cells")
        if not expected_keys.issubset(cartesian_keys):
            raise ValueError("sparse matrix references an unknown goal/distribution")
        coverage = "registered_sparse_cells"
    if set(cell_by_key) != expected_keys:
        if required_cell_keys is None:
            raise ValueError("matrix cells must cover the complete Cartesian product")
        raise ValueError("matrix cells must cover the exact registered sparse set")

    summarized: list[dict[str, Any]] = []
    for distribution_id in sorted(distribution_by_id):
        distribution = distribution_by_id[distribution_id]
        expected_ids = set(distribution.candidate_ids)
        for goal_id in sorted(
            goal_id
            for candidate_distribution_id, goal_id in expected_keys
            if candidate_distribution_id == distribution_id
        ):
            cell = cell_by_key[(distribution_id, goal_id)]
            first = dict(cell.first_labels)
            repeat = dict(cell.repeat_labels)
            if set(first) != expected_ids or set(repeat) != expected_ids:
                raise ValueError(
                    f"matrix cell {distribution_id}/{goal_id} does not cover "
                    "the frozen confirmation cohort exactly"
                )
            stable_counts = {
                label: sum(
                    first[candidate_id] == label
                    and repeat[candidate_id] == label
                    for candidate_id in expected_ids
                )
                for label in ("positive", "negative", "abstain")
            }
            agreement_count = sum(stable_counts.values())
            sample_count = len(expected_ids)
            summarized.append(
                {
                    "distribution_id": distribution_id,
                    "goal_id": goal_id,
                    "sample_count": sample_count,
                    "first_counts": _label_counts(first),
                    "repeat_counts": _label_counts(repeat),
                    "stable_counts": stable_counts,
                    "unstable_count": sample_count - agreement_count,
                    "exact_agreement_fraction": agreement_count / sample_count,
                    "stable_positive_fraction_all": stable_counts["positive"]
                    / sample_count,
                    "first_judgment_artifact_sha256": (
                        cell.first_judgment_artifact_sha256
                    ),
                    "repeat_judgment_artifact_sha256": (
                        cell.repeat_judgment_artifact_sha256
                    ),
                }
            )

    matrix_rows = []
    for distribution_id in sorted(distribution_by_id):
        values = {
            row["goal_id"]: row["stable_positive_fraction_all"]
            for row in summarized
            if row["distribution_id"] == distribution_id
        }
        matrix_rows.append({"distribution_id": distribution_id, "values": values})
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": MATRIX_PROTOCOL,
        "coverage": coverage,
        "registered_cell_keys": [
            {"distribution_id": distribution_id, "goal_id": goal_id}
            for distribution_id, goal_id in sorted(expected_keys)
        ],
        "categorical_labels_only": True,
        "control_loop_input": False,
        "goals": [goal_by_id[key].as_dict() for key in sorted(goal_by_id)],
        "distributions": [
            distribution_by_id[key].as_dict() for key in sorted(distribution_by_id)
        ],
        "cells": summarized,
        "stable_positive_fraction_matrix": matrix_rows,
    }
    artifact["matrix_content_sha256"] = _content_sha256(artifact)
    return artifact


def write_goal_distribution_matrix(path: Path, matrix: Mapping[str, Any]) -> None:
    expected_hash = matrix.get("matrix_content_sha256")
    _validate_sha256(expected_hash, name="matrix_content_sha256")
    content = dict(matrix)
    del content["matrix_content_sha256"]
    if _content_sha256(content) != expected_hash:
        raise ValueError("goal-distribution matrix content hash differs")
    write_immutable_json(path, matrix)
