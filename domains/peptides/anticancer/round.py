"""Run one immutable GPU transition for the V12/V13 moving-frontier protocols.

This entry point is deliberately separate from the historical V6/V12 runner.
It trains on one 100-candidate, current-generator-only alignment split, then
draws matched parent and child cohorts in V12.  V13 reuses its registered
100-candidate current working cohort as the parent diagnostic and draws only a
fresh 100-candidate child cohort.  Qualitative
developability cards are projected onto the one p0 reference frozen before
the campaign; later samples never redefine their own evidence scale.

The process is crash-resumable at the checkpoint, raw-sampling, evidence, and
final-summary boundaries.  Every re-entry accepts only byte-identical inputs
and hash-matching artifacts.
"""

from __future__ import annotations
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Any, Mapping, Sequence
import torch
from domains.peptides.frozen_evidence import apply_frozen_reference
from domains.peptides.model import PepDfmAssets
from domains.peptides.dpo import (
    FLOW_DPO_KIND,
    FlowDPOConfig,
    write_or_validate_pair_manifest,
)
from domains.peptides.checkpoint import (
    PepDfmCheckpoint,
    encode_peptide,
    sha256_file,
    sha256_state_dict,
)
from domains.peptides.pool_io import _predict_private_rows
from domains.peptides.training import (
    all_prior_label_sequences,
    cumulative_training_inputs,
    discover_pending_round,
    immutable_write,
    persist_validated_inputs,
    read_json,
    read_jsonl,
    resolve_parent_checkpoint,
    sample_or_recover_raw_rows,
    train_or_recover_checkpoint_pair,
    validate_execution_authorization,
    validate_round_buffer,
)
from domains.peptides.metrics import jensen_shannon, summarize_samples

PROTOCOL_VERSION = "moving-semantic-frontier-v12d"
GEOMETRIC_PROTOCOL_VERSION = "geometric-semantic-reachability-v13"
EXPECTED_LENGTH_COUNTS = {8: 175, 10: 175, 12: 175, 16: 175}
ALIGNMENT_CANDIDATE_COUNT = 100
MAXIMUM_UPDATES = 10
EXTENDED_MAXIMUM_UPDATES = 40
FROZEN_REFERENCE_METHOD = "p0_length_stratified_tie_aware_empirical_tertiles"
HEX_64 = re.compile("[0-9a-f]{64}")


def _is_geometric_v13(config: Mapping[str, Any]) -> bool:
    return (
        config.get("prompt_protocol", {}).get("version") == GEOMETRIC_PROTOCOL_VERSION
    )


def _maximum_alignment_rounds(config: Mapping[str, Any]) -> int:
    """Return the registered GPU-round ceiling for this protocol.

    Historical V12D campaigns are frozen at ten rounds.  Geometric V13 has
    two preregistered envelopes: the original ten-round campaigns and the
    G01--G05 campaigns with at most eight updates per goal (forty total).
    The low-level GPU runner must agree with the high-level host controller;
    otherwise it could stop early or train beyond the registered budget.
    """
    loop = config.get("curriculum_loop")
    if not isinstance(loop, Mapping):
        raise TypeError("curriculum_loop must be an object")
    observed = _strict_int(
        loop.get("maximum_alignment_rounds"), name="maximum_alignment_rounds"
    )
    if not _is_geometric_v13(config):
        if observed != MAXIMUM_UPDATES:
            raise ValueError(
                "moving-frontier V12D campaign is registered for ten updates"
            )
        return observed
    control = config.get("geometric_control")
    if not isinstance(control, Mapping):
        raise TypeError("geometric_control must be an object for geometric V13")
    registered = _strict_int(
        control.get("maximum_total_updates"), name="maximum_total_updates"
    )
    if registered not in {MAXIMUM_UPDATES, EXTENDED_MAXIMUM_UPDATES}:
        raise ValueError(
            "geometric V13 supports only registered 10- or 40-update envelopes"
        )
    if observed != registered:
        raise ValueError(
            "maximum_alignment_rounds must equal geometric_control.maximum_total_updates"
        )
    return observed


def _expected_length_counts(config: Mapping[str, Any]) -> dict[int, int]:
    """Return the frozen per-distribution length policy.

    V12D keeps its historical four-by-175 contract byte-for-byte.  V13 binds a
    controller-selected policy after the policy call and before p0 sampling;
    every parent/child comparison then reuses that exact policy so a length
    mixture change cannot masquerade as fine-tuning progress.
    """
    if not _is_geometric_v13(config):
        return dict(EXPECTED_LENGTH_COUNTS)
    sampling = config.get("fresh_sampling")
    if not isinstance(sampling, Mapping):
        raise TypeError("fresh_sampling must be an object")
    raw = sampling.get("length_counts_per_arm")
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError("V13 requires a non-empty frozen length policy")
    counts = {
        int(length): _strict_int(count, name=f"length_counts_per_arm[{length}]")
        for length, count in raw.items()
    }
    if not 1 <= len(counts) <= 4:
        raise ValueError("V13 length policy must use one to four exact lengths")
    if any((not 6 <= length <= 49 for length in counts)):
        raise ValueError("V13 peptide lengths must lie in the supported range 6..49")
    if any((not 10 <= count <= 60 for count in counts.values())):
        raise ValueError("V13 per-length counts must lie in 10..60")
    if sum(counts.values()) != 100:
        raise ValueError("V13 length policy must sum to 100 candidates")
    return dict(sorted(counts.items()))


def _expected_candidate_count(config: Mapping[str, Any]) -> int:
    return sum(_expected_length_counts(config).values())


def _approved_initial_parent_checkpoint(
    config: Mapping[str, Any], *, pending_index: int
) -> tuple[Path | None, str | None, str | None]:
    """Return a registered external parent only for extension round zero.

    Ordinary moving-frontier campaigns deliberately start from released
    PepDFM and therefore omit this record.  A separately registered post-hoc
    continuation may instead bind round zero to the terminal checkpoint of a
    completed source experiment.  The source identity and both checkpoint
    hashes are part of the frozen config; later rounds must continue to use
    the preceding child checkpoint and cannot fall back to this source.
    """
    value = config.get("approved_initial_parent_checkpoint")
    if value is None:
        return (None, None, None)
    if not isinstance(value, Mapping):
        raise TypeError("approved_initial_parent_checkpoint must be an object")
    required = {
        "kind": "post_hoc_completed_campaign_extension",
        "source_experiment_id": "EXP-0084",
        "source_generator_id": "p9",
    }
    if any((value.get(name) != expected for name, expected in required.items())):
        raise ValueError("approved initial-parent source identity differs")
    path = Path(str(value.get("path", ""))).resolve()
    expected_sha256 = str(value.get("sha256", "")).lower()
    expected_state_sha256 = str(value.get("state_dict_sha256", "")).lower()
    source_manifest = Path(str(value.get("source_distribution_manifest", ""))).resolve()
    source_manifest_sha256 = str(
        value.get("source_distribution_manifest_sha256", "")
    ).lower()
    for name, observed in {
        "checkpoint sha256": expected_sha256,
        "checkpoint state-dict sha256": expected_state_sha256,
        "source distribution manifest sha256": source_manifest_sha256,
    }.items():
        if HEX_64.fullmatch(observed) is None:
            raise ValueError(f"approved initial-parent {name} is malformed")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError("approved initial-parent checkpoint hash differs")
    if (
        not source_manifest.is_file()
        or sha256_file(source_manifest) != source_manifest_sha256
    ):
        raise ValueError("approved initial-parent distribution binding differs")
    source = read_json(source_manifest)
    checkpoint = source.get("generator_provenance", {}).get("child_checkpoint", {})
    if (
        source.get("distribution_id") != "p9"
        or checkpoint.get("path") != str(path)
        or checkpoint.get("file_sha256") != expected_sha256
        or (checkpoint.get("state_dict_sha256") != expected_state_sha256)
    ):
        raise ValueError("approved initial-parent checkpoint provenance differs")
    if pending_index != 0:
        return (None, None, None)
    return (path, expected_sha256, expected_state_sha256)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json(dict(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _frozen_reference_spec(config: Mapping[str, Any]) -> tuple[Path, str]:
    sampling = config.get("fresh_sampling")
    if not isinstance(sampling, Mapping):
        raise TypeError("fresh_sampling must be an object")
    value = sampling.get("frozen_evidence_reference")
    if not isinstance(value, Mapping):
        raise TypeError("fresh_sampling.frozen_evidence_reference must be an object")
    path = Path(os.path.expandvars(str(value.get("path", "")))).resolve()
    expected_hash = str(value.get("sha256", "")).lower()
    if not path.is_file():
        raise FileNotFoundError(f"frozen p0 evidence reference is missing: {path}")
    if HEX_64.fullmatch(expected_hash) is None:
        raise ValueError("frozen p0 evidence reference requires a SHA-256")
    if sha256_file(path) != expected_hash:
        raise ValueError("frozen p0 evidence reference SHA-256 differs")
    reference = read_json(path)
    if reference.get("method") != FROZEN_REFERENCE_METHOD:
        raise ValueError("frozen p0 evidence reference method differs")
    if reference.get("reference_distribution") != "released unconditional PepDFM p0":
        raise ValueError("evidence bands are not anchored to released p0")
    if reference.get("cardinal_values_private") is not True:
        raise ValueError("frozen reference does not keep cardinal values private")
    expected_groups = {str(length) for length in _expected_length_counts(config)}
    if set(reference.get("groups", {})) != expected_groups:
        raise ValueError("frozen reference does not cover the registered lengths")
    return (path, expected_hash)


def _validated_p0_pool(
    config: Mapping[str, Any], *, reference_sha256: str
) -> tuple[Path, set[str]]:
    """Validate the immutable source pool before any optimization."""
    value = config.get("candidate_pool")
    if not isinstance(value, Mapping):
        raise TypeError("candidate_pool must bind the immutable p0 pool")
    path = Path(os.path.expandvars(str(value.get("path", "")))).resolve()
    expected_hash = str(value.get("sha256", "")).lower()
    if not path.is_file() or HEX_64.fullmatch(expected_hash) is None:
        raise FileNotFoundError("immutable p0 candidate pool binding is incomplete")
    if sha256_file(path) != expected_hash:
        raise ValueError("immutable p0 candidate pool SHA-256 differs")
    rows = read_jsonl(path)
    expected_count = _expected_candidate_count(config)
    if len(rows) != expected_count or value.get("candidate_count") != expected_count:
        raise ValueError(
            f"immutable p0 candidate pool must contain exactly {expected_count} rows"
        )
    ids = [str(row.get("candidate_id", "")) for row in rows]
    sequences = [str(row.get("sequence", "")) for row in rows]
    if any((not item for item in ids + sequences)):
        raise ValueError("immutable p0 candidate pool contains an empty ID or sequence")
    if len(set(ids)) != expected_count or len(set(sequences)) != expected_count:
        raise ValueError("immutable p0 candidate IDs and sequences must be unique")
    observed_lengths = Counter((len(sequence) for sequence in sequences))
    expected_lengths = _expected_length_counts(config)
    if dict(observed_lengths) != expected_lengths:
        raise ValueError("immutable p0 candidate pool differs from its length policy")
    configured_lengths = {
        int(length): _strict_int(count, name=f"candidate_pool.length_counts[{length}]")
        for length, count in dict(value.get("length_counts", {})).items()
    }
    if configured_lengths != expected_lengths:
        raise ValueError("candidate_pool.length_counts differs from the frozen p0 pool")
    manifest_path = path.parent / "pool_manifest.json"
    manifest = read_json(manifest_path)
    expected_manifest = {
        "experiment_id": config["experiment_id"],
        "generator_id": "p0",
        "sample_count": expected_count,
        "candidate_pool_sha256": expected_hash,
        "frozen_evidence_reference_sha256": reference_sha256,
    }
    for name, expected in expected_manifest.items():
        observed = manifest.get(name)
        if name == "generator_id" and observed in {"p0", "p00"}:
            continue
        if observed != expected:
            raise ValueError(f"p0 pool manifest field differs: {name}")
    return (path, set(sequences))


def _all_prior_distribution_sequences(
    output_root: Path, *, through_index: int, expected_count: int
) -> tuple[set[str], list[dict[str, Any]]]:
    """Load every canonical p0..p_k pool with immutable provenance checks."""
    if through_index < 0:
        raise ValueError("through_index must be non-negative")
    all_sequences: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for index in range(through_index + 1):
        distribution_id = f"p{index}"
        manifest_path = (
            output_root
            / "distributions"
            / distribution_id
            / "distribution_split_manifest.json"
        )
        manifest = read_json(manifest_path)
        if manifest.get("distribution_id") != distribution_id:
            raise ValueError("prior distribution manifest identity differs")
        source = manifest.get("source_artifact")
        if not isinstance(source, Mapping):
            raise ValueError("prior distribution manifest omitted its source artifact")
        source_path = Path(str(source.get("path", ""))).resolve()
        source_hash = str(source.get("sha256", ""))
        if not source_path.is_file() or sha256_file(source_path) != source_hash:
            raise ValueError("prior distribution source artifact hash differs")
        rows = read_jsonl(source_path)
        sequences = [
            str(row.get("sequence", "")).strip().upper()
            for row in rows
            if row.get("is_valid", True) is True
        ]
        if (
            len(rows) != expected_count
            or len(sequences) != expected_count
            or len(set(sequences)) != expected_count
        ):
            raise ValueError(
                "each prior distribution must contain the registered number of unique valid peptides"
            )
        overlap = all_sequences.intersection(sequences)
        if overlap:
            raise ValueError("prior canonical distribution pools overlap by sequence")
        all_sequences.update(sequences)
        bindings.append(
            {
                "distribution_id": distribution_id,
                "split_manifest_path": str(manifest_path.resolve()),
                "split_manifest_sha256": sha256_file(manifest_path),
                "source_artifact_path": str(source_path),
                "source_artifact_sha256": source_hash,
                "sequence_count": len(sequences),
            }
        )
    return (all_sequences, bindings)


def validate_moving_frontier_contract(config: Mapping[str, Any]) -> None:
    """Reject scientific drift from the pre-registered GPU transition."""
    if (
        config.get("schema_version") != 1
        or not str(config.get("experiment_id", "")).strip()
    ):
        raise ValueError("moving-frontier runner requires a registered experiment")
    prompt = config.get("prompt_protocol")
    if not isinstance(prompt, Mapping) or prompt.get("version") not in {
        PROTOCOL_VERSION,
        GEOMETRIC_PROTOCOL_VERSION,
    }:
        raise ValueError("unsupported moving-frontier prompt protocol")
    if prompt.get("cardinal_predictor_outputs_visible") is not False:
        raise ValueError("cardinal predictor values must remain private")
    loop = config.get("curriculum_loop")
    maximum_alignment_rounds = _maximum_alignment_rounds(config)
    if maximum_alignment_rounds <= 0:
        raise ValueError("maximum_alignment_rounds must be positive")
    if _strict_int(loop.get("round_seed_stride"), name="round_seed_stride") <= 0:
        raise ValueError("round_seed_stride must be positive")
    discovery = config.get("curriculum_discovery")
    if not isinstance(discovery, Mapping):
        raise TypeError("curriculum_discovery must be an object")
    if (
        _strict_int(discovery.get("sample_size"), name="sample_size")
        != ALIGNMENT_CANDIDATE_COUNT
    ):
        raise ValueError("each alignment split must contain exactly 100 candidates")
    if (
        _strict_int(
            discovery.get("training_label_passes", 2), name="training_label_passes"
        )
        != 2
    ):
        raise ValueError("alignment labels require two independent shuffled passes")
    tuning = config.get("fine_tuning")
    if not isinstance(tuning, Mapping):
        raise TypeError("fine_tuning must be an object")
    replay = config.get("alignment_replay")
    if replay is not None:
        if not _is_geometric_v13(config) or not isinstance(replay, Mapping):
            raise ValueError("cumulative alignment replay requires geometric V13")
        mode = replay.get("mode")
        expected_replay = (
            {
                "mode": "cumulative_same_fixed_goal",
                "fixed_goal_id": "G04",
                "include_all_prior_stable_labels": True,
                "exclude_cross_round_sequence_conflicts": True,
            }
            if mode == "cumulative_same_fixed_goal"
            else {
                "mode": "cumulative_within_current_goal",
                "include_all_prior_stable_labels_at_same_goal": True,
                "reset_on_goal_transition": True,
                "exclude_cross_round_sequence_conflicts": True,
            }
        )
        if mode not in {
            "cumulative_same_fixed_goal",
            "cumulative_within_current_goal",
        } or any(
            (replay.get(name) != value for name, value in expected_replay.items())
        ):
            raise ValueError("cumulative alignment-replay contract differs")
    kind = tuning.get("kind")
    expected_tuning = {"holdout_fraction": 0.0, "steps": 100, "negative_alpha": 0.0}
    if kind != FLOW_DPO_KIND:
        raise ValueError("unsupported moving-frontier fine-tuning kind")
    for name, expected in expected_tuning.items():
        observed = tuning.get(name)
        if isinstance(expected, float):
            try:
                matches = math.isclose(float(observed), expected, abs_tol=0.0)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(f"fine_tuning.{name} must remain {expected!r}")
    expected_dpo = {
        "beta": 2.6,
        "reference_policy": "parent_checkpoint_per_update",
        "pair_construction": "full_cartesian_stable_positive_x_stable_negative",
        "pair_schedule": "deterministic_shuffled_epochs_without_replacement",
        "corruption_coupling": "same_policy_reference_corruption_per_sequence",
        "loss_kind": "diffusion_dpo_logistic_on_generalized_kl_surrogate",
    }
    if any((tuning.get(name) != value for name, value in expected_dpo.items())):
        raise ValueError("flow-DPO fine-tuning contract differs")
    sampling = config.get("fresh_sampling")
    if not isinstance(sampling, Mapping):
        raise TypeError("fresh_sampling must be an object")
    if _strict_int(
        sampling.get("num_samples_per_arm"), name="num_samples_per_arm"
    ) != _expected_candidate_count(config):
        raise ValueError(
            f"the registered diagnostic/child cohort size differs from {_expected_candidate_count(config)}"
        )
    counts = {
        int(length): _strict_int(count, name=f"length_counts_per_arm[{length}]")
        for length, count in dict(sampling.get("length_counts_per_arm", {})).items()
    }
    if counts != _expected_length_counts(config):
        raise ValueError("fresh sampling differs from the frozen length policy")
    if _strict_int(sampling.get("num_steps"), name="fresh_sampling.num_steps") != 100:
        raise ValueError("PepDFM fresh sampling must use 100 flow steps")
    if sampling.get("evidence_banding_reference") != "frozen_p0":
        raise ValueError("later evidence must use the frozen p0 reference")
    _frozen_reference_spec(config)
    gates = config.get("checkpoint_gates")
    if not isinstance(gates, Mapping):
        raise TypeError("checkpoint_gates must be an object")
    if (
        gates.get("require_positive_holdout_loss_decrease") is not False
        or gates.get("require_negative_minus_positive_margin_non_decrease") is not False
    ):
        raise ValueError("all-label updates use non-collapse gates, not heldout losses")
    if float(gates.get("maximum_exact_replay_match_fraction", -1.0)) != 0.0:
        raise ValueError("moving-frontier samples may not replay labelled sequences")
    if (
        float(gates.get("maximum_overlap_with_prior_distribution_fraction", -1.0))
        != 0.0
    ):
        raise ValueError("moving-frontier distributions must be sequence-disjoint")


def moving_sampling_config(
    config: Mapping[str, Any], *, round_index: int
) -> dict[str, Any]:
    """Return the exact deterministic registered per-arm sampling contract."""
    validate_moving_frontier_contract(config)
    values = dict(config["fresh_sampling"])
    values.pop("frozen_evidence_reference")
    values.pop("evidence_banding_reference")
    values.pop("evaluation_samples_per_arm", None)
    count = values.pop("num_samples_per_arm")
    values["num_samples"] = count
    values["seed"] = int(values["seed"]) + round_index * int(
        config["curriculum_loop"]["round_seed_stride"]
    )
    return values


def _validate_alignment_source(
    *,
    buffer: Mapping[str, Any],
    pending_index: int,
    config: Mapping[str, Any] | None = None,
) -> None:
    source = buffer.get("candidate_source")
    if not isinstance(source, Mapping):
        raise ValueError("alignment buffer omitted immutable split provenance")
    expected_generator = f"p{pending_index}"
    if source.get("kind") not in {
        "immutable_alignment_train_split",
        "immutable_working_cohort",
    }:
        raise ValueError("alignment buffer is not an immutable current cohort")
    if source.get("source_generator_id") != expected_generator:
        raise ValueError("alignment buffer is not current-generator-only")
    split_hash = str(source.get("split_manifest_sha256", "")).lower()
    if HEX_64.fullmatch(split_hash) is None:
        raise ValueError("alignment buffer omitted split-manifest provenance")
    if buffer.get("generator_transition") != f"p{pending_index}->p{pending_index + 1}":
        raise ValueError("alignment buffer generator transition differs")
    if len(buffer.get("construction_candidate_ids", ())) != ALIGNMENT_CANDIDATE_COUNT:
        raise ValueError("alignment buffer must bind all 100 train-split candidates")
    if buffer.get("abstentions_and_disagreements_excluded") is not True:
        raise ValueError("unstable/abstaining labels must be excluded from training")
    if config is not None and config["fine_tuning"]["kind"] == FLOW_DPO_KIND:
        if buffer.get("negative_sequences_used_for_dpo_pairs") is not True:
            raise ValueError("stable negatives must enter flow-DPO pairs")
    elif buffer.get("negative_sequences_used_for_dpo_pairs") is not True:
        raise ValueError("stable negatives must enter the DPO update")


def _visible_candidate_id(*, round_index: int, arm: str, position: int) -> str:
    distribution_index = round_index if arm == "parent" else round_index + 1
    return f"p{distribution_index}-t{round_index:02d}-{arm}-{position:04d}"


def _registered_current_parent_rows(
    *, output_root: Path, distribution_index: int, expected_count: int
) -> list[dict[str, Any]]:
    """Recover the one semantic working cohort as V13's parent diagnostic.

    The GPU update must not generate a second parent cohort.  We convert the
    already hash-bound visible rows back to the minimal raw representation
    needed by the frozen-evidence stage; the child remains freshly sampled.
    """
    manifest_path = (
        output_root
        / "distributions"
        / f"p{distribution_index}"
        / "distribution_split_manifest.json"
    )
    manifest = read_json(manifest_path)
    source = manifest.get("source_artifact")
    if (
        manifest.get("distribution_id") != f"p{distribution_index}"
        or manifest.get("candidate_count") != expected_count
        or (not isinstance(source, Mapping))
    ):
        raise ValueError("current working-cohort manifest differs")
    path = Path(str(source.get("path", ""))).resolve()
    if (
        not path.is_file()
        or source.get("sha256") != sha256_file(path)
        or source.get("row_count") != expected_count
    ):
        raise ValueError("current working-cohort source binding differs")
    rows = read_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError("current working cohort has the wrong row count")
    raw: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        sequence = str(row.get("sequence", "")).strip().upper()
        if row.get("is_valid", True) is not True or not sequence:
            raise ValueError("current parent diagnostic contains an invalid peptide")
        raw.append(
            {
                "attempt_index": position,
                "sample_index": position,
                "sequence": sequence,
                "length": len(sequence),
                "token_ids": list(encode_peptide(sequence)),
                "is_valid": True,
                "invalid_reason": None,
                "source": "registered_current_working_cohort",
            }
        )
    return raw


def _moving_noncollapse_gates(
    *, config: Mapping[str, Any], summaries: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, bool], float | None, float | None]:
    """Apply cumulative diversity/composition gates to immutable p0.

    Immediate-parent ratios remain descriptive diagnostics.  The registered
    ``vs_frozen`` thresholds are intentionally evaluated against p0 so several
    individually small updates cannot silently accumulate into collapse.
    """
    gates = config["checkpoint_gates"]
    child = summaries["child"]
    parent = summaries["parent"]
    frozen = summaries["frozen_p0"]
    child_diversity = child["mean_pairwise_normalized_hamming_unique"]
    parent_diversity = parent["mean_pairwise_normalized_hamming_unique"]
    frozen_diversity = frozen["mean_pairwise_normalized_hamming_unique"]
    incremental_ratio = (
        None
        if child_diversity is None or parent_diversity is None or parent_diversity <= 0
        else child_diversity / parent_diversity
    )
    frozen_ratio = (
        None
        if child_diversity is None or frozen_diversity is None or frozen_diversity <= 0
        else child_diversity / frozen_diversity
    )
    values = {
        "validity": child["valid_fraction"] >= float(gates["minimum_valid_fraction"]),
        "unique_valid": child["unique_valid_fraction_of_attempts"]
        >= float(gates["minimum_unique_valid_fraction"]),
        "replay": child["exact_replay_match_fraction"] is not None
        and child["exact_replay_match_fraction"]
        <= float(gates["maximum_exact_replay_match_fraction"]),
        "novelty": child["novel_vs_original_pool_fraction"] is not None
        and child["novel_vs_original_pool_fraction"]
        >= float(gates["minimum_novel_vs_original_pool_fraction"]),
        "diversity_vs_frozen_p0": frozen_ratio is not None
        and frozen_ratio >= float(gates["minimum_diversity_ratio_vs_frozen"]),
        "amino_acid_jsd_vs_frozen_p0": child["amino_acid_jsd_vs_frozen_p0"]
        <= float(gates["maximum_amino_acid_jsd_vs_frozen"]),
    }
    return (values, incremental_ratio, frozen_ratio)


def _visible_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Drop raw token IDs and sampling internals from an API-facing row."""
    value = {
        "candidate_id": str(row["candidate_id"]),
        "generator_id": str(row["generator_id"]),
        "sequence": row.get("sequence"),
        "length": int(row["length"]),
        "is_valid": bool(row.get("is_valid")),
        "invalid_reason": row.get("invalid_reason"),
        "developability_evidence": dict(row.get("developability_evidence", {})),
    }
    if value["is_valid"]:
        if not isinstance(value["sequence"], str) or not value["sequence"]:
            raise ValueError("valid visible candidate omitted its peptide sequence")
        if set(value["developability_evidence"]) != {
            "non_hemolysis",
            "non_fouling",
            "solubility",
            "half_life",
        }:
            raise ValueError("valid visible candidate omitted a frozen evidence band")
    return value


def _frozen_band_histograms(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for arm, rows in rows_by_arm.items():
        by_length: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        missing = 0
        for row in rows:
            evidence = row.get("developability_evidence")
            if not isinstance(evidence, Mapping) or not evidence:
                missing += 1
                continue
            for name, band in evidence.items():
                by_length[str(int(row["length"]))][str(name)][str(band)] += 1
        values[arm] = {
            "missing_evidence_count": missing,
            "by_length": {
                length: {
                    name: dict(sorted(counts.items()))
                    for name, counts in sorted(properties.items())
                }
                for length, properties in sorted(by_length.items())
            },
        }
    return values


def annotate_or_recover_frozen_evidence(
    *,
    raw_rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    sampling: Mapping[str, Any],
    training_root: Path,
    asset_dir: Path,
    device: str,
    round_index: int,
    reference_path: Path,
    reference_sha256: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    """Project raw predictor values onto p0 bands and persist every layer."""
    if set(raw_rows_by_arm) != {"parent", "child"}:
        raise ValueError("frozen evidence requires parent and child arms")
    if sha256_file(reference_path) != reference_sha256:
        raise ValueError("frozen evidence reference changed before annotation")
    reference = read_json(reference_path)
    if reference.get("method") != FROZEN_REFERENCE_METHOD:
        raise ValueError("unsupported frozen evidence reference")
    expected_count = int(sampling["num_samples"])
    raw_hashes = {
        arm: _rows_sha256(rows) for arm, rows in sorted(raw_rows_by_arm.items())
    }
    stage_path = training_root / "evidence_annotation" / "stage.json"
    private_path = training_root / "private" / "predictor_scores.jsonl"
    visible_paths = {
        arm: training_root / "samples" / f"{arm}.jsonl" for arm in ("parent", "child")
    }
    reference_record = {
        "path": str(reference_path.resolve()),
        "sha256": reference_sha256,
        "method": FROZEN_REFERENCE_METHOD,
        "fitted_on": "immutable released-PepDFM p0 pool only",
        "later_distribution_refit": False,
    }
    if stage_path.is_file():
        stage = read_json(stage_path)
        if stage.get("sampling_config") != dict(sampling):
            raise FileExistsError("frozen-evidence sampling config differs")
        if stage.get("raw_rows_sha256") != raw_hashes:
            raise FileExistsError("frozen-evidence raw inputs differ")
        if stage.get("frozen_evidence_reference") != reference_record:
            raise FileExistsError("frozen-evidence reference binding differs")
        rows_by_arm: dict[str, list[dict[str, Any]]] = {}
        for arm, path in visible_paths.items():
            record = stage.get("visible_sample_files", {}).get(arm, {})
            if (
                record.get("sha256") != sha256_file(path)
                or record.get("row_count") != expected_count
            ):
                raise FileExistsError("visible sample differs from its stage receipt")
            rows = read_jsonl(path)
            if len(rows) != expected_count:
                raise ValueError("visible sample row count differs")
            rows_by_arm[arm] = rows
        private_record = stage.get("private_predictor_scores")
        if not isinstance(private_record, Mapping) or private_record.get(
            "sha256"
        ) != sha256_file(private_path):
            raise FileExistsError("private predictor rows differ from stage receipt")
        return (rows_by_arm, stage, dict(private_record))
    indexed: list[dict[str, Any]] = []
    enriched = {
        arm: [dict(row) for row in rows] for arm, rows in raw_rows_by_arm.items()
    }
    for arm in ("parent", "child"):
        if len(enriched[arm]) != expected_count:
            raise ValueError("raw arm row count differs from registered sampling")
        for position, row in enumerate(enriched[arm]):
            candidate_id = _visible_candidate_id(
                round_index=round_index, arm=arm, position=position
            )
            row["candidate_id"] = candidate_id
            row["generator_id"] = (
                f"p{round_index}" if arm == "parent" else f"p{round_index + 1}"
            )
            if row.get("is_valid") is not True:
                continue
            sequence = str(row.get("sequence", "")).strip().upper()
            token_ids = row.get("token_ids")
            if not sequence or not isinstance(token_ids, (list, tuple)):
                raise ValueError("valid raw samples require a sequence and token IDs")
            row["length"] = len(sequence)
            indexed.append(
                {
                    "candidate_id": candidate_id,
                    "sequence": sequence,
                    "length": len(sequence),
                    "token_ids": list(token_ids),
                }
            )
    expected_lengths = {
        int(length): int(count)
        for length, count in dict(sampling["length_counts_per_arm"]).items()
    }
    for arm, rows in enriched.items():
        observed = Counter((int(row.get("length", -1)) for row in rows))
        if dict(observed) != expected_lengths:
            raise ValueError(f"{arm} sample differs from the registered length policy")
    private_core = _predict_private_rows(
        indexed,
        asset_dir=asset_dir,
        device=device,
        batch_size=int(sampling["predictor_batch_size"]),
    )
    metadata = {
        str(row["candidate_id"]): {
            "arm": arm,
            "position": position,
            "generator_id": enriched[arm][position]["generator_id"],
        }
        for arm in ("parent", "child")
        for position, row in enumerate(enriched[arm])
        if row.get("is_valid") is True
    }
    private_rows = [
        dict(row) | metadata[str(row["candidate_id"])] for row in private_core
    ]
    evidence = apply_frozen_reference(private_rows, reference)
    by_id = {
        str(row["candidate_id"]): row
        for arm in ("parent", "child")
        for row in enriched[arm]
    }
    for candidate_id, card in evidence.items():
        by_id[candidate_id]["developability_evidence"] = card
    private_text = "".join((canonical_json(row) + "\n" for row in private_rows))
    immutable_write(private_path, private_text)
    private_record = {
        "path": str(private_path.resolve()),
        "sha256": sha256_file(private_path),
        "row_count": len(private_rows),
        "cardinal_values_visible_to_llm": False,
    }
    visible_rows = {
        arm: [_visible_row(row) for row in enriched[arm]] for arm in ("parent", "child")
    }
    visible_files: dict[str, Any] = {}
    for arm, path in visible_paths.items():
        text = "".join((canonical_json(row) + "\n" for row in visible_rows[arm]))
        immutable_write(path, text)
        visible_files[arm] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "row_count": len(visible_rows[arm]),
        }
    stage = {
        "schema_version": 1,
        "status": "complete",
        "round_index": round_index,
        "sampling_config": dict(sampling),
        "raw_rows_sha256": raw_hashes,
        "frozen_evidence_reference": reference_record,
        "private_predictor_scores": private_record,
        "visible_sample_files": visible_files,
        "visible_values_are_qualitative_bands_only": True,
        "frozen_evidence_band_histograms": _frozen_band_histograms(visible_rows),
    }
    immutable_write(
        stage_path, json.dumps(stage, allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return (visible_rows, stage, private_record)


def execute(
    *, config_path: Path, output_root: Path, asset_dir: Path, campaign_path: Path
) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    execution_authorization = validate_execution_authorization(
        config_path=config_path,
        output_root=output_root,
        campaign_path=campaign_path,
        repository_root=repository_root,
    )
    config = read_json(config_path)
    validate_moving_frontier_contract(config)
    reference_path, reference_hash = _frozen_reference_spec(config)
    _p0_path, p0_sequences = _validated_p0_pool(config, reference_sha256=reference_hash)
    pending = discover_pending_round(
        output_root, maximum_rounds=_maximum_alignment_rounds(config)
    )
    initial_parent_path, initial_parent_sha256, initial_parent_state_sha256 = (
        _approved_initial_parent_checkpoint(config, pending_index=pending.index)
    )
    parent_spec = resolve_parent_checkpoint(
        pending,
        experiment_id=str(config["experiment_id"]),
        explicit_path=initial_parent_path,
        explicit_sha256=initial_parent_sha256,
    )
    buffer = read_json(pending.buffer_path)
    _validate_alignment_source(
        config=config, buffer=buffer, pending_index=pending.index
    )
    validated = validate_round_buffer(
        config=config, config_path=config_path, pending=pending, parent=parent_spec
    )
    if validated["candidate_count"] != ALIGNMENT_CANDIDATE_COUNT:
        raise ValueError("validated alignment buffer does not contain exactly 100 rows")
    replay = config.get("alignment_replay")
    if replay is not None:
        replay_mode = str(replay["mode"])
        current_buffer = read_json(pending.buffer_path)
        fixed_goal_id = str(
            replay.get("fixed_goal_id", current_buffer.get("training_goal_id", ""))
        )
        fixed_goal_artifact = read_json(output_root / "goals" / f"{fixed_goal_id}.json")
        fixed_goal_record = dict(fixed_goal_artifact.get("goal", {}))
        fixed_goal_record["text"] = fixed_goal_record.get("frontier_criterion")
        if (
            fixed_goal_artifact.get("goal_id") != fixed_goal_id
            or not fixed_goal_record.get("frontier_criterion")
            or current_buffer.get("training_goal_id") != fixed_goal_id
            or (current_buffer.get("training_goal") != fixed_goal_record)
        ):
            raise ValueError("frozen cumulative goal artifact differs")
        validated = cumulative_training_inputs(
            pending=pending,
            validated=validated,
            expected_lengths=frozenset(_expected_length_counts(config)),
            strict_fixed_goal_provenance=True,
            strict_experiment_id=str(config["experiment_id"]),
            strict_goal_id=fixed_goal_id,
            strict_training_goal=fixed_goal_record,
            contiguous_same_goal_only=replay_mode == "cumulative_within_current_goal",
        )
        replay_record = validated["buffer_ablation"]
        validated["alignment_policy"] = {
            "mode": replay_mode,
            "fixed_goal_id": fixed_goal_id,
            "source_generator_ids": [
                f"p{row['round_index']}" for row in replay_record["source_buffers"]
            ],
            "current_candidate_count": ALIGNMENT_CANDIDATE_COUNT,
            "previous_round_labels_replayed": len(replay_record["source_buffers"]) > 1,
            "reset_on_goal_transition": replay_mode == "cumulative_within_current_goal",
            "stable_positive_and_negative_labels_used": True,
            "cumulative_positive_count": replay_record["positive_count"],
            "cumulative_negative_count": replay_record["negative_count"],
            "cross_round_conflicts_excluded": len(
                replay_record["cross_round_conflicts_excluded"]
            ),
        }
    else:
        validated["alignment_policy"] = {
            "mode": "current_generator_only",
            "source_generator_id": f"p{pending.index}",
            "candidate_count": ALIGNMENT_CANDIDATE_COUNT,
            "previous_round_labels_replayed": False,
            "stable_positive_and_negative_labels_used": True,
        }
    training_root = pending.training_root
    pair_manifest_path: Path | None = None
    if config["fine_tuning"].get("kind") == FLOW_DPO_KIND:
        pair_manifest_path = training_root / "preference_pairs.jsonl"
        validated["preference_pair_manifest"] = write_or_validate_pair_manifest(
            pair_manifest_path,
            positive_sequences=validated["positive_train"],
            negative_sequences=validated["negative_train"],
        )
    persist_validated_inputs(
        training_root,
        validated=validated,
        execution_authorization=execution_authorization,
    )
    validated["execution_authorization"] = execution_authorization
    if not torch.cuda.is_available():
        raise EnvironmentError("moving-frontier PepDFM update requires a CUDA node")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assets = PepDfmAssets.resolve(asset_dir)
    tuning = config["fine_tuning"]
    parent = PepDfmCheckpoint(assets, device=str(tuning["device"]))
    if parent_spec.path is None:
        parent_load = {
            "path": str(assets.checkpoint),
            "file_sha256": sha256_file(assets.checkpoint),
            "state_dict_sha256": sha256_state_dict(parent.model),
            "kind": "released_pepdfm",
        }
    else:
        assert parent_spec.expected_sha256 is not None
        parent_load = parent.load_checkpoint(
            parent_spec.path, expected_sha256=parent_spec.expected_sha256
        ).as_dict()
        parent_load["kind"] = parent_spec.kind
        if (
            pending.index == 0
            and initial_parent_state_sha256 is not None
            and (parent_load.get("state_dict_sha256") != initial_parent_state_sha256)
        ):
            raise ValueError(
                "loaded initial-parent state-dict hash differs from its approval"
            )
    child = parent.clone()
    stride = int(config["curriculum_loop"]["round_seed_stride"])
    training_seed = int(tuning["seed"]) + pending.index * stride
    train_config = FlowDPOConfig(
        learning_rate=float(tuning["learning_rate"]),
        steps=100,
        batch_size=int(tuning["batch_size"]),
        beta=float(tuning["beta"]),
        gradient_clip_norm=float(tuning["gradient_clip_norm"]),
        epsilon=float(tuning["epsilon"]),
        seed=training_seed,
        reference_policy=str(tuning["reference_policy"]),
        pair_construction=str(tuning["pair_construction"]),
        pair_schedule=str(tuning["pair_schedule"]),
        corruption_coupling=str(tuning["corruption_coupling"]),
        loss_kind=str(tuning["loss_kind"]),
    )
    trace, checkpoint_artifacts = train_or_recover_checkpoint_pair(
        parent=parent,
        child=child,
        training_root=training_root,
        positive_train=validated["positive_train"],
        negative_train=validated["negative_train"],
        train_config=train_config,
        pair_manifest_path=pair_manifest_path,
        round_index=pending.index,
        execution_authorization=execution_authorization,
        legacy_recovery_binding=None,
    )
    sampling = moving_sampling_config(config, round_index=pending.index)
    precomputed_parent = (
        {
            "parent": _registered_current_parent_rows(
                output_root=output_root,
                distribution_index=pending.index,
                expected_count=_expected_candidate_count(config),
            )
        }
        if _is_geometric_v13(config)
        else None
    )
    raw_rows_by_arm, raw_stage = sample_or_recover_raw_rows(
        models={"parent": parent, "child": child},
        sampling=sampling,
        training_root=training_root,
        checkpoint_artifacts=checkpoint_artifacts,
        precomputed_rows_by_arm=precomputed_parent,
    )
    rows_by_arm, evidence_stage, private_artifact = annotate_or_recover_frozen_evidence(
        raw_rows_by_arm=raw_rows_by_arm,
        sampling=sampling,
        training_root=training_root,
        asset_dir=asset_dir,
        device=str(tuning["device"]),
        round_index=pending.index,
        reference_path=reference_path,
        reference_sha256=reference_hash,
    )
    replay_sequences = all_prior_label_sequences(
        output_root, through_round=pending.index
    )
    summaries = {
        arm: summarize_samples(
            rows,
            all_replay_sequences=replay_sequences,
            source_pool_sequences=p0_sequences,
        )
        for arm, rows in rows_by_arm.items()
    }
    p0_rows = [
        {"sequence": sequence, "is_valid": True} for sequence in sorted(p0_sequences)
    ]
    summaries["frozen_p0"] = summarize_samples(
        p0_rows,
        all_replay_sequences=replay_sequences,
        source_pool_sequences=p0_sequences,
    )
    summaries["child"]["amino_acid_jsd_vs_parent"] = jensen_shannon(
        summaries["child"]["amino_acid_distribution"],
        summaries["parent"]["amino_acid_distribution"],
    )
    for arm in ("parent", "child"):
        summaries[arm]["amino_acid_jsd_vs_frozen_p0"] = jensen_shannon(
            summaries[arm]["amino_acid_distribution"],
            summaries["frozen_p0"]["amino_acid_distribution"],
        )
    gate_values, diversity_ratio, diversity_ratio_vs_frozen = _moving_noncollapse_gates(
        config=config, summaries=summaries
    )
    gate_values["checkpoint_state_changed"] = (
        not bool(config["checkpoint_gates"]["require_checkpoint_state_change"])
        or checkpoint_artifacts["parent"]["state_dict_sha256"]
        != checkpoint_artifacts["child"]["state_dict_sha256"]
    )
    prior_distribution_sequences, prior_distribution_bindings = (
        _all_prior_distribution_sequences(
            output_root,
            through_index=pending.index,
            expected_count=_expected_candidate_count(config),
        )
    )
    child_sequences = {
        str(row.get("sequence", "")).strip().upper()
        for row in rows_by_arm["child"]
        if row.get("is_valid") is True and str(row.get("sequence", "")).strip()
    }
    prior_overlap = sorted(child_sequences.intersection(prior_distribution_sequences))
    prior_overlap_fraction = (
        1.0 if not child_sequences else len(prior_overlap) / len(child_sequences)
    )
    summaries["child"].update(
        {
            "exact_overlap_with_prior_distribution_count": len(prior_overlap),
            "exact_overlap_with_prior_distribution_fraction": prior_overlap_fraction,
            "prior_distribution_sequence_count": len(prior_distribution_sequences),
        }
    )
    gate_values["zero_overlap_with_prior_distributions"] = (
        prior_overlap_fraction
        <= float(
            config["checkpoint_gates"][
                "maximum_overlap_with_prior_distribution_fraction"
            ]
        )
    )
    exact_count_key = (
        "exact_100_candidate_cohort_eligibility"
        if _is_geometric_v13(config)
        else "exact_700_candidate_split_eligibility"
    )
    expected_count = _expected_candidate_count(config)
    gate_values[exact_count_key] = (
        summaries["child"]["valid_count"] == expected_count
        and summaries["child"]["unique_valid_count"] == expected_count
    )
    result = {
        "schema_version": 1,
        "protocol_version": str(config["prompt_protocol"]["version"]),
        "experiment_id": config["experiment_id"],
        "round_index": pending.index,
        "generator_transition": f"p{pending.index}->p{pending.index + 1}",
        "claim_boundary": config["claim_boundary"],
        "validated_inputs": validated,
        "parent_checkpoint_load": parent_load,
        "training_trace": trace,
        "checkpoint_sha256": {
            arm: value["file_sha256"] for arm, value in checkpoint_artifacts.items()
        },
        "checkpoint_state_dict_sha256": {
            arm: value["state_dict_sha256"]
            for arm, value in checkpoint_artifacts.items()
        },
        "child_checkpoint_artifact": {
            "path": str((training_root / "checkpoints" / "child.pt").resolve()),
            "file_sha256": checkpoint_artifacts["child"]["file_sha256"],
            "state_dict_sha256": checkpoint_artifacts["child"]["state_dict_sha256"],
            "generator_id": f"p{pending.index + 1}",
        },
        "sample_summaries": summaries,
        "raw_sample_files": raw_stage["sample_files"],
        "visible_sample_files": evidence_stage["visible_sample_files"],
        "child_distribution_artifact": {
            **evidence_stage["visible_sample_files"]["child"],
            "generator_id": f"p{pending.index + 1}",
            "representation": "candidate_id, generator_id, sequence, length, validity, and frozen-p0 qualitative developability evidence",
        },
        "private_predictor_scores": private_artifact,
        "frozen_evidence_reference": evidence_stage["frozen_evidence_reference"],
        "frozen_evidence_band_histograms": evidence_stage[
            "frozen_evidence_band_histograms"
        ],
        "fresh_sampling_length_counts_per_arm": {
            str(length): count
            for length, count in _expected_length_counts(config).items()
        },
        "prior_distribution_bindings": prior_distribution_bindings,
        "prior_distribution_overlap": {
            "count": len(prior_overlap),
            "fraction_of_unique_valid_child": prior_overlap_fraction,
            "maximum_registered_fraction": float(
                config["checkpoint_gates"][
                    "maximum_overlap_with_prior_distribution_fraction"
                ]
            ),
            "overlap_sequence_sha256": hashlib.sha256(
                canonical_json(prior_overlap).encode("utf-8")
            ).hexdigest(),
            "sequences_not_persisted_in_summary": True,
        },
        "training_seed": training_seed,
        "matched_sampling_seed": int(sampling["seed"]),
        "same_source_randomness_across_parent_and_child": not _is_geometric_v13(config),
        "parent_diagnostic_source": "registered_current_100_peptide_working_cohort; no additional parent sampling"
        if _is_geometric_v13(config)
        else "fresh matched parent sample",
        "diversity_ratio_child_vs_parent": diversity_ratio,
        "diversity_ratio_child_vs_frozen_p0": diversity_ratio_vs_frozen,
        "checkpoint_gate": {
            "conditions": gate_values,
            "all_conditions_met": all(gate_values.values()),
            "child_may_seed_next_round_only_if_all_conditions_met": True,
            "scientific_goal_movement_is_a_separate_heldout_cpu_llm_gate": True,
        },
        "resumability": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in {
                "checkpoint_stage": training_root / "checkpoints" / "stage.json",
                **(
                    {"preference_pair_manifest": pair_manifest_path}
                    if pair_manifest_path is not None
                    else {}
                ),
                "raw_sampling_stage": training_root / "raw_sampling" / "stage.json",
                "evidence_annotation_stage": training_root
                / "evidence_annotation"
                / "stage.json",
            }.items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
        },
    }
    immutable_write(
        training_root / "summary.json",
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--active-campaign", type=Path, required=True)
    args = parser.parse_args()
    result = execute(
        config_path=args.config.resolve(),
        output_root=args.output_root.resolve(),
        asset_dir=args.asset_dir.resolve(),
        campaign_path=args.active_campaign.resolve(),
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
