"""Run the V13 geometric semantic-reachability campaign.

One immutable 100-peptide working cohort represents every generator state.
The same cohort is used to select a temporary goal, obtain two categorical
label passes, and construct that state's alignment buffer.  No independently
generated goal-validation cohort and no LLM cardinal score exists in this
protocol.  A fresh child cohort becomes the next state only after the existing
GPU transition runner passes its non-collapse checks.
"""

from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Literal, Mapping, Sequence
from domains.peptides.reachability import (
    MAXIMUM_PROPOSAL_ATTEMPTS,
    audit_learnability,
    categorical_report_by_length,
    choose_after_update,
    correction_request,
    length_policy_matches_campaign,
)
from adm.preferences.allocation_window import (
    AllocationYieldRequired,
    require_allocation_window,
)
from adm.preferences.artifacts import FilePreferenceArtifactStore
from adm.preferences.gemini_vertex import GeminiUsageBudget, GeminiVertexClient
from adm.preferences.provider import bind_provider, make_structured_client, provider_key, without_credentials
from domains.peptides.controller import (
    GeometricFrontierController,
    GeometricFrontierProposal,
)
from domains.peptides.teacher import (
    MOVING_FRONTIER_AUDITOR_PROMPT,
    MovingFrontierProposal,
    MovingFrontierSemanticAuditor,
)
from domains.peptides.audit import (
    QMatrixCell,
    audit_initial_broadening,
    build_q_matrix_cell,
    stratified_bootstrap_rate_difference,
)
from domains.peptides.sequence_evidence import (
    frozen_p0_developability_evidence_tool_definition,
)
from domains.peptides.goals import (
    GEOMETRIC_V13_JUDGE_SYSTEM_PROMPT,
    GoalCorrectionRequest,
)
from domains.peptides.length_policy import load_pre_p0_length_policy_artifact
from domains.peptides.types import PeptideCandidate, SemanticLabel
from domains.peptides.judgments import (
    SupportLevel,
    V4CandidateJudge,
    V4JudgeResult,
    V4SampleJudgment,
)
from domains.peptides.frontier import FrontierGoal, frontier_as_dict
from adm.artifacts import (
    bind_model_version,
    canonical_json,
    immutable_json,
    immutable_jsonl,
    immutable_write,
    model_version_binding_records,
    read_json,
    read_jsonl,
    sha256_file,
    sync_agent_model_version_binding,
)
from adm.campaign import resolve_campaign_registration

PROTOCOL_VERSION = "geometric-semantic-reachability-v13"
FLOW_DPO_KIND = "all_cartesian_pairs_pepdfm_flow_dpo"
ANTICANCER_FINAL_GOAL = "Prefer peptides that appear more promising as selective direct anticancer therapeutics: plausibly active against cancer cells while minimizing nonspecific toxicity toward healthy human cells."
FINAL_GOAL_ID = "F"
TRANSITION_COUNT_METRIC = "completed_geometric_semantic_transition_count"
EXACT_FINAL_PRIMARY_METRIC = "exact_final_goal_newly_reachable_after_generator_updates"
ROLE_NAMES = ("candidate_judge", "curriculum_controller", "independent_auditor")
SHA256_RE = re.compile("^[0-9a-f]{64}$")


@dataclass(frozen=True)
class AgentBundle:
    judge: V4CandidateJudge
    controller: GeometricFrontierController
    auditor: MovingFrontierSemanticAuditor


@dataclass(frozen=True)
class JudgmentPair:
    first: V4JudgeResult
    repeat: V4JudgeResult
    cell: QMatrixCell
    root: Path


@dataclass(frozen=True)
class AcceptedProposal:
    proposal: GeometricFrontierProposal
    pair: JudgmentPair
    entry_pair: JudgmentPair | None
    proposal_path: Path
    decision_path: Path


def _path_from_config(value: object) -> Path:
    return Path(os.path.expandvars(str(value))).resolve()


def _require_hash_bound_file(binding: Mapping[str, Any], *, name: str) -> Path:
    path = _path_from_config(binding.get("path", ""))
    expected = str(binding.get("sha256", "")).lower()
    if not path.is_file() or SHA256_RE.fullmatch(expected) is None:
        raise FileNotFoundError(f"{name} is not an immutable hash-bound file")
    if sha256_file(path) != expected:
        raise ValueError(f"{name} SHA-256 differs")
    return path


def _authorized_length_policy_selection_method(config: Mapping[str, Any]) -> str:
    """Return the campaign resource contract for the experiment's length policy."""
    return "one Gemini controller call under the registered bounded schema"


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _at_least(observed: float, threshold: float) -> bool:
    return observed > threshold or math.isclose(
        observed, threshold, rel_tol=0.0, abs_tol=1e-12
    )


def _campaign_length_counts(config: Mapping[str, Any]) -> dict[int, int]:
    raw = config.get("fresh_sampling", {}).get("length_counts_per_arm")
    if not isinstance(raw, Mapping) or not raw:
        raise TypeError("fresh_sampling.length_counts_per_arm must be non-empty")
    result = dict(
        sorted(
            (
                (int(key), _strict_int(value, name="length count"))
                for key, value in raw.items()
            )
        )
    )
    if not 1 <= len(result) <= 4:
        raise ValueError("campaign length policy requires one to four lengths")
    if any((not 6 <= length <= 49 for length in result)):
        raise ValueError("campaign peptide lengths must lie in [6, 49]")
    if any((not 10 <= count <= 60 for count in result.values())):
        raise ValueError("campaign per-length counts must lie in [10, 60]")
    if sum(result.values()) != 100:
        raise ValueError("campaign length counts must sum to 100")
    return result


def conservative_candidate_judge_call_bound(config: Mapping[str, Any]) -> int:
    """Return the exact registered worst-case number of judge API calls.

    One categorical cell uses two complete passes.  Each proposal is judged on
    the current cohort; tightening proposals are also judged on the stored
    goal-entry cohort.  Before every possible tightening, exact F is tested on
    those same two distributions.  Campaigns with an explicit intermediate-
    goal ceiling also train the last permitted goal and test exact F once more
    before terminating, without ever proposing a goal beyond that ceiling.
    """
    sample_count = _strict_int(
        config["curriculum_discovery"]["sample_size"], name="sample_size"
    )
    chunk_size = _strict_int(config["candidate_judge"]["chunk_size"], name="chunk_size")
    calls_per_cell = 2 * math.ceil(sample_count / chunk_size)
    attempts = _strict_int(
        config["geometric_control"]["maximum_proposal_attempts"],
        name="maximum_proposal_attempts",
    )
    updates = _strict_int(
        config["geometric_control"]["maximum_total_updates"],
        name="maximum_total_updates",
    )
    control = config["geometric_control"]
    if "maximum_intermediate_goals" in control:
        intermediate_goals = _strict_int(
            control["maximum_intermediate_goals"], name="maximum_intermediate_goals"
        )
        if intermediate_goals < 1:
            raise ValueError("maximum_intermediate_goals must be positive")
        cell_count = (
            1
            + attempts
            + updates
            + 2 * intermediate_goals
            + 2 * attempts * (intermediate_goals - 1)
        )
    else:
        tightenings = _strict_int(
            control["maximum_semantic_tightenings"], name="maximum_semantic_tightenings"
        )
        cell_count = 1 + attempts + updates + tightenings * (2 + 2 * attempts)
    return calls_per_cell * cell_count


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the complete CPU-control contract used by this runner."""
    if (
        config.get("schema_version") != 1
        or not str(config.get("experiment_id", "")).strip()
    ):
        raise ValueError("V13 requires schema_version 1 and an experiment_id")
    if not str(config.get("final_goal", "")).strip():
        raise ValueError("the immutable final_goal must be non-empty")
    prompt = config.get("prompt_protocol")
    if (
        not isinstance(prompt, Mapping)
        or prompt.get("version") != PROTOCOL_VERSION
        or prompt.get("cardinal_predictor_outputs_visible") is not False
        or (not isinstance(prompt.get("experiment_context"), Mapping))
    ):
        raise ValueError("V13 prompt protocol or context differs")
    expected_prompts = {
        "candidate_judge_prompt": "GEOMETRIC_V13_JUDGE_SYSTEM_PROMPT",
        "controller_prompt": "GEOMETRIC_V13_CONTROLLER_PROMPT plus GEOMETRIC_V13_CORRECTION_ADDENDUM on correction attempts",
        "independent_auditor_prompt": "MOVING_FRONTIER_AUDITOR_PROMPT",
        "pre_p0_length_prompt": "PRE_P0_LENGTH_POLICY_SYSTEM_PROMPT",
    }
    if (
        config.get("final_goal") != ANTICANCER_FINAL_GOAL
        or any((prompt.get(key) != value for key, value in expected_prompts.items()))
        or prompt.get("task_family") is not None
    ):
        raise ValueError("anticancer task or prompt binding differs")
    discovery = config.get("curriculum_discovery")
    expected_discovery = {
        "sample_size": 100,
        "training_label_passes": 2,
        "minimum_stable_positive_count": 20,
        "minimum_stable_negative_count": 20,
        "minimum_exact_agreement": 0.5,
        "maximum_abstain_fraction_per_pass": 0.5,
    }
    if not isinstance(discovery, Mapping) or any(
        (discovery.get(key) != value for key, value in expected_discovery.items())
    ):
        raise ValueError("V13 categorical learnability contract differs")
    control = config.get("geometric_control")
    expected_control = {
        "maximum_proposal_attempts": 4,
        "maximum_corrections": 3,
        "minimum_enrichment": 0.1,
        "maximum_updates_per_goal": 5,
        "maximum_total_updates": 10,
        "minimum_initial_broadening_gain": 0.1,
        "minimum_enriched_positive_fraction": 0.6,
        "minimum_new_goal_unlock": 0.1,
        "maximum_semantic_tightenings": 3,
    }
    if not isinstance(control, Mapping) or any(
        (control.get(key) != value for key, value in expected_control.items())
    ):
        raise ValueError("V13 geometric host-control constants differ")
    loop = config.get("curriculum_loop")
    expected_rounds = 10
    if (
        not isinstance(loop, Mapping)
        or loop.get("maximum_alignment_rounds") != expected_rounds
    ):
        raise ValueError(
            f"GPU transition loop must permit exactly {expected_rounds} updates"
        )
    evaluation = config.get("evaluation")
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("independent_generated_validation_cohort") is not False
        or evaluation.get("working_cohort_size") != 100
        or (evaluation.get("cardinal_scores_used") is not False)
    ):
        raise ValueError("V13 one-cohort categorical evaluation contract differs")
    lengths = _campaign_length_counts(config)
    policy_binding = config.get("campaign_length_policy")
    if not isinstance(policy_binding, Mapping):
        raise TypeError("campaign_length_policy must be an object")
    policy_path = _require_hash_bound_file(
        policy_binding, name="campaign length policy"
    )
    policy_artifact = load_pre_p0_length_policy_artifact(policy_path)
    policy_lengths = {
        row.length: row.count for row in policy_artifact.length_policy.allocations
    }
    if (
        policy_lengths != lengths
        or policy_binding.get("selected_before_p0") is not True
        or policy_binding.get("frozen_for_entire_campaign") is not True
    ):
        raise ValueError("campaign length-policy artifact or freeze contract differs")
    pool = config.get("candidate_pool")
    if not isinstance(pool, Mapping):
        raise TypeError("candidate_pool must be an object")
    _require_hash_bound_file(pool, name="candidate_pool")
    if pool.get("candidate_count") != 100:
        raise ValueError("candidate_pool must contain exactly 100 candidates")
    pool_lengths = {
        int(key): int(value)
        for key, value in dict(pool.get("length_counts", {})).items()
    }
    if pool_lengths != lengths:
        raise ValueError("candidate_pool length policy differs from fresh sampling")
    if (
        config.get("anticancer_reference_context") is not None
        or prompt.get("labelled_reference_context_visible") is not False
        or prompt.get("external_peptide_dataset_visible") is not False
    ):
        raise ValueError("V13 must not use an external peptide dataset or reference")
    for role_name in ROLE_NAMES:
        role = config.get(role_name)
        if not isinstance(role, Mapping):
            raise TypeError(f"{role_name} must be an object")
        for key in (
            "model",
            "reasoning_effort",
            "max_output_tokens",
            "maximum_provider_calls",
            "api_budget",
        ):
            if key not in role:
                raise ValueError(f"{role_name} omitted {key}")
    judge = config["candidate_judge"]
    if judge.get("chunk_size") not in {20, 25} or not isinstance(
        judge.get("provider_order_seed"), int
    ):
        raise ValueError(
            "candidate_judge must use bounded chunks and a fixed order seed"
        )
    registered_bound = conservative_candidate_judge_call_bound(config)
    aggregate = config.get("aggregate_api_budget")
    if (
        not isinstance(aggregate, Mapping)
        or aggregate.get("conservative_candidate_judge_call_bound") != registered_bound
        or int(judge["maximum_provider_calls"]) < registered_bound
    ):
        raise ValueError("candidate-judge worst-case call bound differs")
    transport = config.get("transport")
    if not isinstance(transport, Mapping) or any(
        (
            key not in transport
            for key in ("request_timeout_seconds", "requests_per_minute_per_role")
        )
    ):
        raise ValueError("transport configuration is incomplete")
    tuning = config.get("fine_tuning")
    if not isinstance(tuning, Mapping):
        raise ValueError("V13 fine-tuning contract is missing")
    kind = tuning.get("kind")
    expected_negative_alpha = 0.0
    if (
        kind != FLOW_DPO_KIND
        or float(tuning.get("negative_alpha", -1.0)) != expected_negative_alpha
        or int(tuning.get("steps", -1)) != 100
        or (float(tuning.get("holdout_fraction", -1.0)) != 0.0)
    ):
        raise ValueError("V13 all-label fine-tuning contract differs")
    expected_dpo = {
        "beta": 2.6,
        "reference_policy": "parent_checkpoint_per_update",
        "pair_construction": "full_cartesian_stable_positive_x_stable_negative",
        "pair_schedule": "deterministic_shuffled_epochs_without_replacement",
        "corruption_coupling": "same_policy_reference_corruption_per_sequence",
        "loss_kind": "diffusion_dpo_logistic_on_generalized_kl_surrogate",
    }
    if any((tuning.get(name) != value for name, value in expected_dpo.items())):
        raise ValueError("V13 flow-DPO contract differs")


def _usage_budget(role: Mapping[str, Any]) -> GeminiUsageBudget:
    value = role["api_budget"]
    return GeminiUsageBudget(
        maximum_prompt_tokens=int(value["maximum_prompt_tokens"]),
        maximum_output_tokens=int(value["maximum_output_tokens"]),
        maximum_estimated_usd=float(value["maximum_estimated_usd"]),
        input_usd_per_million_tokens=float(value["input_usd_per_million_tokens"]),
        output_usd_per_million_tokens=float(value["output_usd_per_million_tokens"]),
        prompt_token_overhead_per_request=int(
            value["prompt_token_overhead_per_request"]
        ),
    )


def _ensure_preflight(output_root: Path, name: str, role: Mapping[str, Any]) -> None:
    path = output_root / "preflight" / f"{name}.json"
    if not path.is_file():
        raise PermissionError(f"missing generateContent preflight receipt for {name}")
    value = read_json(path)
    if (
        value.get("model") != role["model"]
        or value.get("http_status") != 200
        or value.get("generation_requested") is not True
        or (not str(value.get("model_version", "")).strip())
    ):
        raise PermissionError(f"invalid generateContent preflight receipt for {name}")


def build_roles(config: Mapping[str, Any], output_root: Path) -> AgentBundle:
    bind_provider(output_root, legacy_glob="**/attempt_started.json")
    length_policy_path = _require_hash_bound_file(
        config["campaign_length_policy"], name="campaign length policy"
    )
    length_policy = load_pre_p0_length_policy_artifact(length_policy_path).length_policy
    for name in ROLE_NAMES:
        _ensure_preflight(output_root, name, config[name])

    def make_client(name: str) -> tuple[GeminiVertexClient, int]:
        attempts = output_root / "api_attempts" / name
        initial = (
            len(list(attempts.glob("*/attempt_started.json")))
            if attempts.exists()
            else 0
        )
        role = config[name]
        if initial > int(role["maximum_provider_calls"]):
            raise PermissionError(f"{name} durable call attempts exceed its ceiling")
        client = make_structured_client(
            timeout=float(config["transport"]["request_timeout_seconds"]),
            temperature=0.0,
            attempt_root=attempts,
            requests_per_minute=float(
                config["transport"]["requests_per_minute_per_role"]
            ),
            usage_budget=_usage_budget(role),
        )
        return (client, initial)

    judge_client, judge_initial = make_client("candidate_judge")
    controller_client, controller_initial = make_client("curriculum_controller")
    auditor_client, auditor_initial = make_client("independent_auditor")
    experiment_context = {
        **dict(config["prompt_protocol"]["experiment_context"]),
        "external_dataset_or_reference_context_visible": False,
        "target_specific_anticancer_predictor_visible": False,
        "campaign_length_policy_is_frozen_before_p0": {
            str(key): value for key, value in _campaign_length_counts(config).items()
        },
    }
    judge_config = config["candidate_judge"]
    judge = V4CandidateJudge(
        client=judge_client,
        artifact_store=FilePreferenceArtifactStore(
            output_root / "api_artifacts" / "candidate_judge"
        ),
        model=str(judge_config["model"]),
        reasoning_effort=str(judge_config["reasoning_effort"]),
        max_output_tokens=int(judge_config["max_output_tokens"]),
        max_provider_calls=int(judge_config["maximum_provider_calls"]),
        initial_provider_calls=judge_initial,
        chunk_size=int(judge_config["chunk_size"]),
        provider_order_seed=int(judge_config["provider_order_seed"]),
        system_prompt=GEOMETRIC_V13_JUDGE_SYSTEM_PROMPT,
        use_generic_evidence_cards=True,
        experiment_context=experiment_context,
        show_final_goal_context_for_intermediate=True,
        developability_evidence_tool=frozen_p0_developability_evidence_tool_definition(),
        require_model_version=True,
        maximum_coverage_redispatches_per_chunk=int(
            judge_config.get("maximum_coverage_redispatches_per_chunk", 25)
        ),
    )
    controller_config = config["curriculum_controller"]
    controller = GeometricFrontierController(
        client=controller_client,
        artifact_store=FilePreferenceArtifactStore(
            output_root / "api_artifacts" / "curriculum_controller"
        ),
        model=str(controller_config["model"]),
        reasoning_effort=str(controller_config["reasoning_effort"]),
        max_output_tokens=int(controller_config["max_output_tokens"]),
        max_provider_calls=int(controller_config["maximum_provider_calls"]),
        initial_provider_calls=controller_initial,
        campaign_length_policy=length_policy,
        experiment_context=experiment_context,
    )
    auditor_config = config["independent_auditor"]
    auditor = MovingFrontierSemanticAuditor(
        client=auditor_client,
        artifact_store=FilePreferenceArtifactStore(
            output_root / "api_artifacts" / "independent_auditor"
        ),
        model=str(auditor_config["model"]),
        reasoning_effort=str(auditor_config["reasoning_effort"]),
        max_output_tokens=int(auditor_config["max_output_tokens"]),
        max_provider_calls=int(auditor_config["maximum_provider_calls"]),
        initial_provider_calls=auditor_initial,
        judge_mode="dual_goal_v1",
        system_prompt=MOVING_FRONTIER_AUDITOR_PROMPT,
    )
    return AgentBundle(judge, controller, auditor)


def _candidate_from_row(row: Mapping[str, Any]) -> PeptideCandidate:
    evidence = row.get("developability_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("working candidates require qualitative evidence")
    return PeptideCandidate(
        candidate_id=str(row["candidate_id"]),
        sequence=str(row["sequence"]).strip().upper(),
        developability_evidence=tuple(
            sorted(((str(key), str(value)) for key, value in evidence.items()))
        ),
    )


def _validate_candidate_rows(
    rows: Sequence[Mapping[str, Any]], length_counts: Mapping[int, int]
) -> tuple[PeptideCandidate, ...]:
    if len(rows) != 100:
        raise ValueError("each V13 working cohort must have exactly 100 rows")
    candidates = tuple((_candidate_from_row(row) for row in rows))
    if len({row.candidate_id for row in candidates}) != 100:
        raise ValueError("working candidate IDs must be unique")
    if len({row.sequence for row in candidates}) != 100:
        raise ValueError("working candidate sequences must be unique")
    observed = Counter((len(row.sequence) for row in candidates))
    if dict(observed) != dict(length_counts):
        raise ValueError("working cohort differs from the campaign length policy")
    return candidates


def _distribution_manifest_path(output_root: Path, distribution_id: str) -> Path:
    return (
        output_root
        / "distributions"
        / distribution_id
        / "distribution_split_manifest.json"
    )


def ensure_working_distribution(
    *,
    output_root: Path,
    distribution_id: str,
    source_path: Path,
    generator_provenance: Mapping[str, Any],
    length_counts: Mapping[int, int],
) -> Path:
    candidates = _validate_candidate_rows(read_jsonl(source_path), length_counts)
    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "distribution_id": distribution_id,
        "cohort_kind": "single_immutable_working_cohort",
        "candidate_count": 100,
        "length_counts": {str(key): value for key, value in length_counts.items()},
        "source_artifact": {
            "path": str(source_path.resolve()),
            "sha256": sha256_file(source_path),
            "row_count": 100,
        },
        "generator_provenance": dict(generator_provenance),
        "candidate_id_sha256": hashlib.sha256(
            canonical_json([row.candidate_id for row in candidates]).encode("utf-8")
        ).hexdigest(),
        "independent_generated_validation_cohort": False,
    }
    path = _distribution_manifest_path(output_root, distribution_id)
    immutable_json(path, manifest)
    return path


def working_candidates(
    *, output_root: Path, distribution_id: str, length_counts: Mapping[int, int]
) -> tuple[PeptideCandidate, ...]:
    manifest = read_json(_distribution_manifest_path(output_root, distribution_id))
    if (
        manifest.get("distribution_id") != distribution_id
        or manifest.get("candidate_count") != 100
    ):
        raise ValueError("working distribution manifest identity differs")
    source = manifest.get("source_artifact")
    if not isinstance(source, Mapping):
        raise ValueError("working distribution manifest omitted source artifact")
    path = _require_hash_bound_file(source, name=f"{distribution_id} working cohort")
    return _validate_candidate_rows(read_jsonl(path), length_counts)


def _serialize_result(result: V4JudgeResult) -> dict[str, Any]:
    value = frontier_as_dict(result)
    if not isinstance(value, dict):
        raise TypeError("judge-result serialization failed")
    if not result.coverage_redispatch_request_hashes:
        value.pop("coverage_redispatch_request_hashes", None)
    value["counts"] = result.counts()
    return value


def _deserialize_result(value: Mapping[str, Any]) -> V4JudgeResult:
    judgments = value.get("judgments")
    if not isinstance(judgments, list) or not judgments:
        raise ValueError("stored judge result omitted judgments")
    rows = []
    for row in judgments:
        if not isinstance(row, Mapping):
            raise ValueError("stored judgment row is malformed")
        rows.append(
            V4SampleJudgment(
                candidate_id=str(row["candidate_id"]),
                model_label=SemanticLabel(str(row["model_label"])),
                label=SemanticLabel(str(row["label"])),
                goal_specific_support=str(row["goal_specific_support"]),
                cancer_application_support=SupportLevel(
                    str(row["cancer_application_support"])
                ),
                healthy_cell_safety_selectivity_support=SupportLevel(
                    str(row["healthy_cell_safety_selectivity_support"])
                ),
                generic_proxy_only=bool(row["generic_proxy_only"]),
                known_nonspecific_toxicity_concern=bool(
                    row["known_nonspecific_toxicity_concern"]
                ),
                negative_due_only_to_missing_evidence=bool(
                    row["negative_due_only_to_missing_evidence"]
                ),
                contraindications=str(row["contraindications"]),
                critical_unknowns=str(row["critical_unknowns"]),
                rationale=str(row["rationale"]),
                override_reason=None
                if row.get("override_reason") is None
                else str(row["override_reason"]),
            )
        )
    result = V4JudgeResult(
        goal=str(value["goal"]),
        exact_final_goal=bool(value["exact_final_goal"]),
        judgments=tuple(rows),
        request_hashes=tuple((str(item) for item in value.get("request_hashes", []))),
        usage=(),
        model_versions=tuple((str(item) for item in value.get("model_versions", []))),
        thought_summaries=tuple(
            (str(item) for item in value.get("thought_summaries", []))
        ),
        coverage_recovery_request_hashes=tuple(
            (str(item) for item in value.get("coverage_recovery_request_hashes", []))
        ),
        coverage_redispatch_request_hashes=tuple(
            (str(item) for item in value.get("coverage_redispatch_request_hashes", []))
        ),
    )
    return result


def _labels(result: V4JudgeResult) -> dict[str, str]:
    return {row.candidate_id: row.label.value for row in result.judgments}


def _judge_iteration(distribution_id: str, judgment_key: str) -> int:
    digest = hashlib.sha256(
        f"{distribution_id}:{judgment_key}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 2147483647


def _descriptive_difference_interval(
    previous: QMatrixCell, current: QMatrixCell, *, key: str
) -> dict[str, Any]:
    seed = _judge_iteration("bootstrap", key)
    if previous.generator_id == current.generator_id:
        return audit_initial_broadening(
            previous, current, semantic_direction_passed=True
        ).broadening_interval.as_mapping()
    return stratified_bootstrap_rate_difference(
        previous, current, replicates=10000, confidence_level=0.95, seed=seed
    ).as_mapping()


def judge_pair(
    *,
    judge: V4CandidateJudge,
    goal_id: str,
    judgment_key: str,
    goal: FrontierGoal,
    distribution_id: str,
    samples: Sequence[PeptideCandidate],
    output_root: Path,
) -> JudgmentPair:
    if len(samples) != 100:
        raise ValueError("V13 judgment requires exactly 100 candidates")
    root = output_root / "judgments" / distribution_id / judgment_key
    results: list[V4JudgeResult] = []
    for repeat_index in (0, 1):
        sync_agent_model_version_binding(output_root=output_root, agent=judge)
        result = judge.judge_current_goal(
            goal.frontier_criterion,
            samples,
            final_goal=goal.final_goal,
            arm=f"{distribution_id}_{judgment_key}",
            iteration=_judge_iteration(distribution_id, judgment_key),
            repeat_index=repeat_index,
            proxy_specification=goal.operative_proxy_specification(),
        )
        bind_model_version(
            output_root=output_root,
            role_name="candidate_judge",
            requested_model=judge.model,
            observed_versions=result.model_versions,
        )
        immutable_json(
            root / f"pass-{repeat_index}.json",
            {
                "schema_version": 1,
                "distribution_id": distribution_id,
                "goal_id": goal_id,
                "judgment_key": judgment_key,
                "repeat_index": repeat_index,
                "candidate_count": 100,
                "result": _serialize_result(result),
            },
        )
        results.append(result)
    cell = build_q_matrix_cell(
        generator_id=distribution_id,
        goal_id=goal_id,
        first_labels=_labels(results[0]),
        repeat_labels=_labels(results[1]),
        strata={row.candidate_id: str(len(row.sequence)) for row in samples},
    )
    immutable_json(root / "stable_summary.json", cell.as_mapping(include_outcomes=True))
    immutable_json(
        root / "categorical_report.json",
        categorical_report_by_length(candidates=samples, cell=cell),
    )
    return JudgmentPair(results[0], results[1], cell, root)


def load_judgment_cell(path: Path) -> QMatrixCell:
    value = read_json(path)
    labels = value.get("stable_labels", {}).get("outcomes")
    if not isinstance(labels, list) or not labels:
        raise ValueError("stable-summary artifact omitted outcomes")
    first = {str(row["candidate_id"]): str(row["first_label"]) for row in labels}
    repeat = {str(row["candidate_id"]): str(row["repeat_label"]) for row in labels}
    strata = {str(row["candidate_id"]): str(row["stratum"]) for row in labels}
    return build_q_matrix_cell(
        generator_id=str(value["generator_id"]),
        goal_id=str(value["goal_id"]),
        first_labels=first,
        repeat_labels=repeat,
        strata=strata,
    )


def load_judgment_pair(root: Path) -> JudgmentPair:
    first_envelope = read_json(root / "pass-0.json")
    repeat_envelope = read_json(root / "pass-1.json")
    first = _deserialize_result(first_envelope["result"])
    repeat = _deserialize_result(repeat_envelope["result"])
    cell = load_judgment_cell(root / "stable_summary.json")
    expected_ids = {row.candidate_id for row in first.judgments}
    if expected_ids != {
        row.candidate_id for row in repeat.judgments
    } or expected_ids != set(cell.summary.candidate_ids):
        raise ValueError("stored judgment pair candidate coverage differs")
    return JudgmentPair(first=first, repeat=repeat, cell=cell, root=root)


def _goal_path(output_root: Path, goal_id: str) -> Path:
    return output_root / "goals" / f"{goal_id}.json"


def write_goal(
    *, output_root: Path, goal_id: str, goal: FrontierGoal, source: Mapping[str, Any]
) -> Path:
    path = _goal_path(output_root, goal_id)
    immutable_json(
        path,
        {
            "schema_version": 1,
            "goal_id": goal_id,
            "goal": frontier_as_dict(goal),
            "source": dict(source),
            "criterion_sha256": hashlib.sha256(
                goal.frontier_criterion.encode("utf-8")
            ).hexdigest(),
        },
    )
    return path


def load_goal(output_root: Path, goal_id: str) -> FrontierGoal:
    value = read_json(_goal_path(output_root, goal_id))
    goal = value.get("goal")
    if not isinstance(goal, Mapping) or value.get("goal_id") != goal_id:
        raise ValueError("goal artifact is malformed")
    result = FrontierGoal(
        final_goal=str(goal["final_goal"]),
        frontier_criterion=str(goal["frontier_criterion"]),
        bridge_to_final=str(goal["bridge_to_final"]),
        is_final=bool(goal["is_final"]),
        positive_evidence_definition=goal.get("positive_evidence_definition"),
        negative_evidence_definition=goal.get("negative_evidence_definition"),
        positive_witness_sequence=goal.get("positive_witness_sequence"),
        negative_witness_sequence=goal.get("negative_witness_sequence"),
        limitations=str(goal["limitations"]),
    )
    if (goal_id == FINAL_GOAL_ID) is not result.is_final:
        raise ValueError("final/intermediate goal identity differs")
    return result


def _state_paths(output_root: Path) -> list[Path]:
    root = output_root / "state"
    return sorted(root.glob("snapshot-[0-9][0-9][0-9].json")) if root.exists() else []


def load_state(output_root: Path) -> dict[str, Any] | None:
    paths = _state_paths(output_root)
    if not paths:
        return None
    prior: Path | None = None
    result: dict[str, Any] | None = None
    for index, path in enumerate(paths):
        if path.name != f"snapshot-{index:03d}.json":
            raise ValueError("state snapshots are not contiguous")
        value = read_json(path)
        if value.get("snapshot_index") != index:
            raise ValueError("state snapshot index differs")
        expected = None if prior is None else sha256_file(prior)
        if value.get("prior_snapshot_sha256") != expected:
            raise ValueError("state snapshot chain differs")
        prior = path
        result = value
    return result


def write_state(output_root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    paths = _state_paths(output_root)
    result = dict(value)
    result["schema_version"] = 1
    result["snapshot_index"] = len(paths)
    result["prior_snapshot_sha256"] = None if not paths else sha256_file(paths[-1])
    path = output_root / "state" / f"snapshot-{len(paths):03d}.json"
    immutable_json(path, result)
    return result


def allocation_api_attempt_ledger(output_root: Path) -> dict[str, Any]:
    """Describe and hash every durable provider-attempt artifact in a prefix."""
    starts = sorted(output_root.glob("**/attempt_started.json"))
    responses = sorted(output_root.glob("**/provider_response.json"))
    parsed = sorted(output_root.glob("**/parsed_response.json"))
    orphaned = [
        path
        for path in starts
        if not path.with_name("provider_response.json").is_file()
    ]
    responses_without_start = [
        path
        for path in responses
        if not path.with_name("attempt_started.json").is_file()
    ]
    parsed_without_response = [
        path
        for path in parsed
        if not path.with_name("provider_response.json").is_file()
    ]
    if orphaned or responses_without_start or parsed_without_response:
        malformed = {
            "ambiguous_attempts": [
                str(path.relative_to(output_root)) for path in orphaned
            ],
            "responses_without_start": [
                str(path.relative_to(output_root)) for path in responses_without_start
            ],
            "parsed_without_response": [
                str(path.relative_to(output_root)) for path in parsed_without_response
            ],
        }
        raise RuntimeError(f"allocation API attempt ledger is malformed: {malformed}")
    artifacts = [
        {"path": str(path.relative_to(output_root)), "sha256": sha256_file(path)}
        for path in sorted([*starts, *responses, *parsed])
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            artifacts, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    return {
        "started": len(starts),
        "provider_responses": len(responses),
        "parsed_responses": len(parsed),
        "ambiguous_attempts": 0,
        "artifact_count": len(artifacts),
        "artifact_manifest_sha256": manifest_sha256,
    }


def allocation_prefix_artifact_ledger(output_root: Path) -> dict[str, Any]:
    """Hash the complete immutable semantic prefix, excluding yield receipts."""
    artifacts = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(output_root)
        if relative.parts and relative.parts[0] == "allocation_yields":
            continue
        artifacts.append({"path": str(relative), "sha256": sha256_file(path)})
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            artifacts, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    return {
        "artifact_count": len(artifacts),
        "artifact_manifest_sha256": manifest_sha256,
    }


def allocation_registered_input_ledger(
    *, config_path: Path, authorization_path: Path, campaign_path: Path
) -> dict[str, Any]:
    """Hash every immutable input outside the campaign-output prefix."""
    task_root = config_path.parent
    registered: list[tuple[str, Path]] = [
        ("task_output/frozen_config.json", config_path),
        (
            "task_output/frozen_config_manifest.json",
            task_root / "frozen_config_manifest.json",
        ),
        (
            "task_output/campaign_length_policy.json",
            task_root / "campaign_length_policy.json",
        ),
        ("authorization", authorization_path),
        ("active_campaign", campaign_path),
    ]
    for directory_name in ("p0", "smoke"):
        directory = task_root / directory_name
        files = sorted((path for path in directory.rglob("*") if path.is_file()))
        if not files:
            raise FileNotFoundError(
                f"registered task input directory has no files: {directory}"
            )
        registered.extend(
            (
                (f"task_output/{directory_name}/{path.relative_to(directory)}", path)
                for path in files
            )
        )
    missing = [str(path) for _label, path in registered if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"registered allocation inputs are missing: {missing}")
    artifacts = [
        {"label": label, "sha256": sha256_file(path)}
        for label, path in sorted(registered, key=lambda item: item[0])
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(
            artifacts, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()
    return {
        "artifact_count": len(artifacts),
        "artifact_manifest_sha256": manifest_sha256,
    }


def _write_allocation_yield(
    *,
    config_path: Path,
    output_root: Path,
    authorization_path: Path,
    campaign_path: Path,
    error: AllocationYieldRequired,
) -> dict[str, Any]:
    """Bind a cooperative yield to the unchanged resumable prefix."""
    allocation_id = str(os.environ.get("SLURM_JOB_ID", ""))
    if not allocation_id.isdigit():
        raise EnvironmentError("allocation yield requires a numeric SLURM_JOB_ID")
    if any(
        (
            (output_root / name).exists()
            for name in ("final_report.json", "final_validation.json")
        )
    ):
        raise RuntimeError("terminal artifacts cannot publish an allocation yield")
    api_attempt_ledger = allocation_api_attempt_ledger(output_root)
    prefix_artifact_ledger = allocation_prefix_artifact_ledger(output_root)
    registered_input_ledger = allocation_registered_input_ledger(
        config_path=config_path,
        authorization_path=authorization_path,
        campaign_path=campaign_path,
    )
    state_paths = _state_paths(output_root)
    state = load_state(output_root)
    latest_path = state_paths[-1] if state_paths else None
    receipt = {
        "schema_version": 1,
        "status": "allocation_yield",
        "scientific_terminal": False,
        "resume_required": True,
        "allocation_id": allocation_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "insufficient_registered_allocation_window",
        "blocked_operation": error.operation,
        "blocked_request_hash": error.request_hash,
        "deadline_epoch": error.deadline_epoch,
        "remaining_seconds": error.remaining_seconds,
        "required_seconds": error.required_seconds,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "latest_state_snapshot": None
        if latest_path is None
        else {
            "path": str(latest_path.resolve()),
            "sha256": sha256_file(latest_path),
            "snapshot_index": int(state["snapshot_index"]),
            "phase": str(state["phase"]),
            "total_updates": int(state["total_updates"]),
            "current_goal_id": state.get("current_goal_id"),
            "current_distribution_id": state.get("current_distribution_id"),
        },
        "api_attempt_ledger": api_attempt_ledger,
        "prefix_artifact_ledger": prefix_artifact_ledger,
        "registered_input_ledger": registered_input_ledger,
        "semantic_state_was_not_mutated_for_yield": True,
    }
    path = output_root / "allocation_yields" / f"allocation-{allocation_id}.json"
    immutable_json(path, receipt)
    return {
        "status": "allocation_yield",
        "scientific_terminal": False,
        "resume_required": True,
        "allocation_id": allocation_id,
        "receipt_path": str(path.resolve()),
        "receipt_sha256": sha256_file(path),
    }


def _audit_proposal(
    *,
    roles: AgentBundle,
    proposal: GeometricFrontierProposal,
    source_goal: FrontierGoal,
    final_goal: str,
    history: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    adapted = MovingFrontierProposal(
        action=proposal.action,
        goal=proposal.goal,
        raw_response=proposal.raw_response,
        metadata=proposal.metadata,
        positive_witness_candidate_id=proposal.positive_witness_candidate_id,
        negative_witness_candidate_id=proposal.negative_witness_candidate_id,
    )
    sync_agent_model_version_binding(output_root=output_root, agent=roles.auditor)
    result = roles.auditor.audit(
        proposal=adapted,
        source_goal=source_goal,
        final_goal=final_goal,
        goal_history=history,
    )
    bind_model_version(
        output_root=output_root,
        role_name="independent_auditor",
        requested_model=roles.auditor.model,
        observed_versions=(str(result.get("model_version", "")),),
    )
    return result


def _goal_history(output_root: Path, goal_ids: Sequence[str]) -> list[dict[str, Any]]:
    return [read_json(_goal_path(output_root, goal_id)) for goal_id in goal_ids]


def _proposal_name(
    *, action: str, distribution_id: str, goal_number: int, attempt: int
) -> str:
    return f"{distribution_id}-{action}-G{goal_number:02d}-attempt-{attempt}"


def _extra_correction_request(
    *,
    prior: str,
    cell: QMatrixCell,
    attempt: int,
    reason: str,
    representative_outcomes: Sequence[Mapping[str, Any]],
) -> GoalCorrectionRequest:
    base = (
        correction_request(
            proposal_attempt=attempt,
            prior_frontier_criterion=prior,
            cell=cell,
            semantic_direction_passed=True,
            length_policy_passed=True,
            representative_proposed_goal_outcomes=representative_outcomes,
        )
        if not audit_learnability(cell).passed
        else None
    )
    if base is not None:
        return base
    from domains.peptides.reachability import stable_counts

    return GoalCorrectionRequest(
        proposal_attempt=attempt,
        correction_number=attempt - 1,
        prior_frontier_criterion=prior,
        observed_counts=stable_counts(cell),
        representative_proposed_goal_outcomes=list(representative_outcomes),
        failed_conditions=[reason],
        host_revision_direction="narrow_slightly",
        semantic_feedback="The goal was already about as common at goal entry as it is now. Add the smallest biologically meaningful distinction that is rarer at entry while remaining discriminative on the current cohort.",
        outcomes_are_from_same_working_cohort=True,
        exact_class_quota_is_not_the_target=True,
    )


def _initial_gain_correction_request(
    *,
    prior: str,
    cell: QMatrixCell,
    attempt: int,
    representative_outcomes: Sequence[Mapping[str, Any]],
) -> GoalCorrectionRequest:
    from domains.peptides.reachability import stable_counts

    return GoalCorrectionRequest(
        proposal_attempt=attempt,
        correction_number=attempt - 1,
        prior_frontier_criterion=prior,
        observed_counts=stable_counts(cell),
        representative_proposed_goal_outcomes=list(representative_outcomes),
        failed_conditions=["initial_goal_not_materially_broader_than_exact_final"],
        host_revision_direction="broaden_materially",
        semantic_feedback="The proposal increased the stable-positive fraction by less than ten percentage points over the exact final goal on this same p0 cohort. Remove or defer a genuine final-like requirement and choose a weaker scientifically meaningful first rung; do not target a class quota.",
        outcomes_are_from_same_working_cohort=True,
        exact_class_quota_is_not_the_target=True,
    )


def _representative_correction_outcomes(
    samples: Sequence[PeptideCandidate], pair: JudgmentPair
) -> list[dict[str, Any]]:
    """Return six bounded, provider-safe examples from the failed proposal."""
    sorted_samples = sorted(samples, key=lambda row: row.candidate_id)
    item_id = {
        row.candidate_id: f"item-{index:03d}"
        for index, row in enumerate(sorted_samples, 1)
    }
    sequence_by_id = {row.candidate_id: row.sequence for row in samples}
    first = {row.candidate_id: row for row in pair.first.judgments}
    repeat = {row.candidate_id: row for row in pair.repeat.judgments}
    groups: dict[str, list[Any]] = {
        "positive": [],
        "negative": [],
        "ambiguous_or_unstable": [],
    }
    for outcome in pair.cell.summary.outcomes:
        key = (
            outcome.stable_label.value
            if outcome.stable_label.value in {"positive", "negative"}
            else "ambiguous_or_unstable"
        )
        groups[key].append(outcome)
    visible: list[dict[str, Any]] = []
    for key in ("positive", "negative", "ambiguous_or_unstable"):
        for outcome in groups[key][:2]:
            visible.append(
                {
                    "item_id": item_id[outcome.candidate_id],
                    "sequence": sequence_by_id[outcome.candidate_id],
                    "stable_outcome": outcome.stable_label.value,
                    "pass_1_label": outcome.first_label,
                    "pass_2_label": outcome.repeat_label,
                    "pass_1_rationale": first[outcome.candidate_id].rationale[:1200],
                    "pass_2_rationale": repeat[outcome.candidate_id].rationale[:1200],
                    "pass_1_critical_unknowns": first[
                        outcome.candidate_id
                    ].critical_unknowns[:1200],
                    "pass_2_critical_unknowns": repeat[
                        outcome.candidate_id
                    ].critical_unknowns[:1200],
                }
            )
    return visible


def propose_reachable_goal(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    roles: AgentBundle,
    action: Literal["broaden", "tighten"],
    source_goal: FrontierGoal,
    current_distribution_id: str,
    current_samples: Sequence[PeptideCandidate],
    source_pair: JudgmentPair,
    goal_ids: Sequence[str],
    update_count: int,
    goal_entry_distribution_id: str | None = None,
    goal_entry_samples: Sequence[PeptideCandidate] | None = None,
    goal_entry_source_pair: JudgmentPair | None = None,
) -> AcceptedProposal | None:
    """Try one proposal plus at most three informed corrections."""
    goal_number = len([goal_id for goal_id in goal_ids if goal_id != FINAL_GOAL_ID]) + 1
    correction: GoalCorrectionRequest | None = None
    campaign_lengths = _campaign_length_counts(config)
    history = _goal_history(output_root, goal_ids)
    for attempt in range(1, MAXIMUM_PROPOSAL_ATTEMPTS + 1):
        sync_agent_model_version_binding(
            output_root=output_root, agent=roles.controller
        )
        proposal = roles.controller.propose(
            action=action,
            final_goal=source_goal.final_goal,
            current_goal=source_goal,
            current_samples=current_samples,
            current_result=source_pair.first,
            generator_id=current_distribution_id,
            goal_history=history,
            update_count=update_count,
            proposal_attempt=attempt,
            correction=correction,
            goal_entry_samples=goal_entry_samples,
            goal_entry_result=None
            if goal_entry_source_pair is None
            else goal_entry_source_pair.first,
            goal_entry_generator_id=goal_entry_distribution_id,
        )
        bind_model_version(
            output_root=output_root,
            role_name="curriculum_controller",
            requested_model=roles.controller.model,
            observed_versions=(str(proposal.metadata.get("model_version", "")),),
        )
        semantic = _audit_proposal(
            roles=roles,
            proposal=proposal,
            source_goal=source_goal,
            final_goal=source_goal.final_goal,
            history=history,
            output_root=output_root,
        )
        length_passed = length_policy_matches_campaign(
            proposal.length_policy, campaign_lengths
        )
        name = _proposal_name(
            action=action,
            distribution_id=current_distribution_id,
            goal_number=goal_number,
            attempt=attempt,
        )
        proposed_pair = judge_pair(
            judge=roles.judge,
            goal_id=f"G{goal_number:02d}",
            judgment_key=name,
            goal=proposal.goal,
            distribution_id=current_distribution_id,
            samples=current_samples,
            output_root=output_root,
        )
        entry_pair: JudgmentPair | None = None
        newly_reachable = True
        if action == "tighten":
            assert goal_entry_distribution_id is not None
            assert goal_entry_samples is not None
            entry_pair = judge_pair(
                judge=roles.judge,
                goal_id=f"G{goal_number:02d}",
                judgment_key=f"{name}-goal-entry",
                goal=proposal.goal,
                distribution_id=goal_entry_distribution_id,
                samples=goal_entry_samples,
                output_root=output_root,
            )
            newly_reachable = _at_least(
                proposed_pair.cell.q - entry_pair.cell.q,
                float(config["geometric_control"]["minimum_new_goal_unlock"]),
            )
        initial_broadening_gain = proposed_pair.cell.q - source_pair.cell.q
        initial_broadening_passed = action != "broaden" or _at_least(
            initial_broadening_gain,
            float(config["geometric_control"]["minimum_initial_broadening_gain"]),
        )
        learnable = audit_learnability(proposed_pair.cell)
        stable_by_id = proposed_pair.cell.summary.by_candidate_id()
        positive_witness = stable_by_id.get(str(proposal.positive_witness_candidate_id))
        negative_witness = stable_by_id.get(str(proposal.negative_witness_candidate_id))
        witnesses_passed = (
            positive_witness is not None
            and positive_witness.stable_label.value == "positive"
            and (negative_witness is not None)
            and (negative_witness.stable_label.value == "negative")
        )
        accepted = (
            semantic.get("accepted") is True
            and length_passed
            and learnable.passed
            and newly_reachable
            and initial_broadening_passed
            and witnesses_passed
        )
        proposal_path = output_root / "proposals" / f"{name}.json"
        immutable_json(
            proposal_path,
            {
                "schema_version": 1,
                "proposal_name": name,
                "proposal": proposal.as_mapping(),
                "semantic_audit": semantic,
                "campaign_length_policy_repeated_exactly": length_passed,
                "current_categorical_cell": proposed_pair.cell.as_mapping(),
                "goal_entry_categorical_cell": None
                if entry_pair is None
                else entry_pair.cell.as_mapping(),
                "descriptive_95pct_stratified_bootstrap_difference": _descriptive_difference_interval(
                    source_pair.cell if entry_pair is None else entry_pair.cell,
                    proposed_pair.cell,
                    key=name,
                ),
                "newly_reachable_by_ten_percentage_points": newly_reachable,
                "initial_broadening_gain_over_exact_final": initial_broadening_gain
                if action == "broaden"
                else None,
                "initial_broadening_gain_passed": initial_broadening_passed,
                "proposed_witnesses_match_stable_categorical_outcomes": witnesses_passed,
                "learnability": learnable.as_mapping(),
                "accepted": accepted,
                "same_100_candidate_working_cohort_reused": True,
                "cardinal_scores_used": False,
            },
        )
        decision_path = output_root / "decisions" / f"{name}.json"
        immutable_json(
            decision_path,
            {
                "schema_version": 1,
                "proposal_path": str(proposal_path.resolve()),
                "proposal_sha256": sha256_file(proposal_path),
                "accepted": accepted,
                "correction_available": not accepted
                and attempt < MAXIMUM_PROPOSAL_ATTEMPTS,
            },
        )
        if accepted:
            return AcceptedProposal(
                proposal, proposed_pair, entry_pair, proposal_path, decision_path
            )
        if attempt == MAXIMUM_PROPOSAL_ATTEMPTS:
            return None
        representatives = _representative_correction_outcomes(
            current_samples, proposed_pair
        )
        if not initial_broadening_passed:
            correction = _initial_gain_correction_request(
                prior=proposal.goal.frontier_criterion,
                cell=proposed_pair.cell,
                attempt=attempt + 1,
                representative_outcomes=representatives,
            )
        elif (
            not newly_reachable
            and learnable.passed
            and (semantic.get("accepted") is True)
            and length_passed
        ):
            correction = _extra_correction_request(
                prior=proposal.goal.frontier_criterion,
                cell=proposed_pair.cell,
                attempt=attempt + 1,
                reason="tighter_goal_not_newly_reachable",
                representative_outcomes=representatives,
            )
        else:
            correction = correction_request(
                proposal_attempt=attempt + 1,
                prior_frontier_criterion=proposal.goal.frontier_criterion,
                cell=proposed_pair.cell,
                semantic_direction_passed=semantic.get("accepted") is True
                and witnesses_passed,
                length_policy_passed=length_passed,
                representative_proposed_goal_outcomes=representatives,
                complementary_temporary_negatives=False,
            )
    raise AssertionError("bounded proposal loop fell through")


def _stable_rows(
    candidates: Sequence[PeptideCandidate], pair: JudgmentPair
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    sequence_by_id = {row.candidate_id: row.sequence for row in candidates}
    positive: list[dict[str, str]] = []
    negative: list[dict[str, str]] = []
    excluded: list[str] = []
    for outcome in pair.cell.summary.outcomes:
        row = {
            "candidate_id": outcome.candidate_id,
            "sequence": sequence_by_id[outcome.candidate_id],
        }
        if outcome.stable_label.value == "positive":
            positive.append(row)
        elif outcome.stable_label.value == "negative":
            negative.append(row)
        else:
            excluded.append(outcome.candidate_id)
    return (positive, negative, excluded)


def stage_alignment_buffer(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    round_index: int,
    distribution_id: str,
    goal_id: str,
    goal: FrontierGoal,
    candidates: Sequence[PeptideCandidate],
    pair: JudgmentPair,
    semantic_trajectory: Sequence[str],
) -> Path:
    learnability = audit_learnability(pair.cell)
    if not learnability.passed:
        raise ValueError("cannot train from a non-learnable categorical cell")
    round_root = output_root / "rounds" / f"round-{round_index:02d}"
    candidate_path = round_root / "candidates.jsonl"
    immutable_jsonl(
        candidate_path,
        [
            {
                "candidate_id": row.candidate_id,
                "sequence": row.sequence,
                "developability_evidence": dict(row.developability_evidence),
            }
            for row in candidates
        ],
    )
    manifest_path = _distribution_manifest_path(output_root, distribution_id)
    positive, negative, excluded = _stable_rows(candidates, pair)
    stability = pair.cell.summary.as_mapping(include_outcomes=True)
    stability.update(
        {
            "stable_positive_ids": [row["candidate_id"] for row in positive],
            "stable_negative_ids": [row["candidate_id"] for row in negative],
            "unstable_ids": excluded,
            "passed": True,
        }
    )
    buffer = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "round_index": round_index,
        "generator_transition": f"p{round_index}->p{round_index + 1}",
        "candidate_pool_sha256": sha256_file(candidate_path),
        "candidate_source": {
            "kind": "immutable_working_cohort",
            "source_generator_id": distribution_id,
            "split_manifest_path": str(manifest_path.resolve()),
            "split_manifest_sha256": sha256_file(manifest_path),
            "role": "working",
        },
        "construction_candidate_ids": [row.candidate_id for row in candidates],
        "training_goal_id": goal_id,
        "training_goal": {**frontier_as_dict(goal), "text": goal.frontier_criterion},
        "pre_update_counts": pair.first.counts(),
        "repeat_label_stability": stability,
        "stable_positive": positive,
        "stable_negative": negative,
        "abstentions_and_disagreements_excluded": True,
        "fine_tuning_authorized": True,
        "negative_sequences_are_evaluation_only": False,
        "negative_sequences_used_for_dpo_pairs": True,
        "training_objective": "all Cartesian stable-positive/stable-negative flow-DPO",
        "negative_gradient_weight": 0.0,
        "llm_cardinal_scores_used": False,
        "first_judge_request_hashes": list(pair.first.request_hashes),
        "repeat_judge_request_hashes": list(pair.repeat.request_hashes),
        "semantic_trajectory": list(semantic_trajectory),
        "working_cohort_reused_for_goal_selection_labels_and_training": True,
        "independent_generated_validation_cohort": False,
    }
    path = round_root / "alignment_buffer.json"
    immutable_json(path, buffer)
    return path


def _initialize(
    *, config: Mapping[str, Any], config_path: Path, output_root: Path
) -> dict[str, Any]:
    lengths = _campaign_length_counts(config)
    pool_path = _require_hash_bound_file(
        config["candidate_pool"], name="candidate_pool"
    )
    p0_manifest = ensure_working_distribution(
        output_root=output_root,
        distribution_id="p0",
        source_path=pool_path,
        generator_provenance={
            "kind": "released_unconditional_pepdfm",
            "candidate_pool_sha256": sha256_file(pool_path),
            "generator_checkpoint_sha256": config["candidate_pool"].get(
                "generator_checkpoint_sha256"
            ),
        },
        length_counts=lengths,
    )
    final_goal = FrontierGoal.exact_final(str(config["final_goal"]))
    final_path = write_goal(
        output_root=output_root,
        goal_id=FINAL_GOAL_ID,
        goal=final_goal,
        source={
            "kind": "immutable_user_final_goal",
            "experiment_id": config["experiment_id"],
        },
    )
    return write_state(
        output_root,
        {
            "experiment_id": config["experiment_id"],
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "phase": "need_initial_goal",
            "terminal_status": None,
            "terminal_reason": None,
            "goal_ids": [FINAL_GOAL_ID],
            "current_goal_id": FINAL_GOAL_ID,
            "current_distribution_id": "p0",
            "current_goal_entry_distribution_id": "p0",
            "current_goal_entry_cell_path": None,
            "total_updates": 0,
            "updates_at_current_goal": 0,
            "distribution_ids": ["p0"],
            "updates": [],
            "accepted_transitions": [],
            "p0_manifest_path": str(p0_manifest.resolve()),
            "p0_manifest_sha256": sha256_file(p0_manifest),
            "final_goal_path": str(final_path.resolve()),
            "final_goal_sha256": sha256_file(final_path),
            "external_dataset_or_reference_context_visible": False,
            "cardinal_scores_used": False,
            "independent_generated_validation_cohort": False,
        },
    )


def _adopt_and_stage(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    state: Mapping[str, Any],
    distribution_id: str,
    goal_id: str,
    goal: FrontierGoal,
    pair: JudgmentPair,
    goal_entry_cell_path: Path,
    goal_path: Path,
    transition: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidates = working_candidates(
        output_root=output_root,
        distribution_id=distribution_id,
        length_counts=_campaign_length_counts(config),
    )
    stage_alignment_buffer(
        config=config,
        output_root=output_root,
        round_index=int(state["total_updates"]),
        distribution_id=distribution_id,
        goal_id=goal_id,
        goal=goal,
        candidates=candidates,
        pair=pair,
        semantic_trajectory=[
            *state["goal_ids"],
            *([] if goal_id in state["goal_ids"] else [goal_id]),
        ],
    )
    next_state = dict(state)
    if goal_id not in next_state["goal_ids"]:
        next_state["goal_ids"] = [*next_state["goal_ids"], goal_id]
    if transition is not None:
        next_state["accepted_transitions"] = [
            *next_state["accepted_transitions"],
            dict(transition),
        ]
    next_state.update(
        {
            "phase": "ready_to_train",
            "current_goal_id": goal_id,
            "current_goal_entry_distribution_id": distribution_id,
            "current_goal_entry_cell_path": str(goal_entry_cell_path.resolve()),
            "updates_at_current_goal": 0,
            "current_goal_path": str(goal_path.resolve()),
            "pending_train_cell_path": str(
                (pair.root / "stable_summary.json").resolve()
            ),
        }
    )
    return write_state(output_root, next_state)


def discover_initial_goal(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    roles: AgentBundle,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    samples = working_candidates(
        output_root=output_root,
        distribution_id="p0",
        length_counts=_campaign_length_counts(config),
    )
    final_goal = load_goal(output_root, FINAL_GOAL_ID)
    final_pair = judge_pair(
        judge=roles.judge,
        goal_id=FINAL_GOAL_ID,
        judgment_key="initial-exact-final",
        goal=final_goal,
        distribution_id="p0",
        samples=samples,
        output_root=output_root,
    )
    final_learnability = audit_learnability(final_pair.cell)
    if final_learnability.passed:
        next_state = dict(state)
        next_state["accepted_transitions"] = [
            *state["accepted_transitions"],
            {
                "kind": "exact_final_trainable_at_p0",
                "target_goal_id": FINAL_GOAL_ID,
                "current_distribution_id": "p0",
                "q_current": final_pair.cell.q,
                "judgment_cell_path": str(
                    (final_pair.root / "stable_summary.json").resolve()
                ),
                "learnability": final_learnability.as_mapping(),
            },
        ]
        next_state.update(
            phase="terminal_complete",
            terminal_status="ineligible_final_goal_reachable_at_p0"
            if _uses_exact_final_primary(config)
            else "reached_final_goal",
            terminal_reason="the exact immutable final goal was already trainable on p0; the protocol stops without fine-tuning",
        )
        return write_state(output_root, next_state)
    accepted = propose_reachable_goal(
        config=config,
        output_root=output_root,
        roles=roles,
        action="broaden",
        source_goal=final_goal,
        current_distribution_id="p0",
        current_samples=samples,
        source_pair=final_pair,
        goal_ids=state["goal_ids"],
        update_count=0,
    )
    if accepted is None:
        next_state = dict(state)
        next_state.update(
            phase="terminal_complete",
            terminal_status="falsified",
            terminal_reason="no learnable initial goal after one proposal and three corrections",
        )
        return write_state(output_root, next_state)
    goal_id = "G01"
    goal_path = write_goal(
        output_root=output_root,
        goal_id=goal_id,
        goal=accepted.proposal.goal,
        source={
            "kind": "accepted_initial_broadening",
            "proposal_path": str(accepted.proposal_path.resolve()),
            "proposal_sha256": sha256_file(accepted.proposal_path),
            "decision_path": str(accepted.decision_path.resolve()),
            "decision_sha256": sha256_file(accepted.decision_path),
        },
    )
    return _adopt_and_stage(
        config=config,
        output_root=output_root,
        state=state,
        distribution_id="p0",
        goal_id=goal_id,
        goal=accepted.proposal.goal,
        pair=accepted.pair,
        goal_entry_cell_path=accepted.pair.root / "stable_summary.json",
        goal_path=goal_path,
        transition={
            "kind": "initial_broadening",
            "source_goal_id": FINAL_GOAL_ID,
            "target_goal_id": goal_id,
            "distribution_id": "p0",
            "proposal_path": str(accepted.proposal_path.resolve()),
        },
    )


def execute_training_update(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    campaign_path: Path,
    asset_dir: Path,
    pepdfm_python: Path,
    dependency_overlay: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    round_index = int(state["total_updates"])
    if round_index >= int(config["geometric_control"]["maximum_total_updates"]):
        raise PermissionError("V13 maximum total updates reached")
    summary_path = (
        output_root
        / "rounds"
        / f"round-{round_index:02d}"
        / "training"
        / "summary.json"
    )
    if not summary_path.is_file():
        require_allocation_window(
            required_seconds=3600.0, operation=f"gpu_transition_round_{round_index:02d}"
        )
        environment = dict(os.environ)
        environment = without_credentials(environment)
        repository_root = Path(__file__).resolve().parents[3]
        environment["PYTHONPATH"] = f"{repository_root}:{dependency_overlay}" + (
            f":{environment['PYTHONPATH']}" if environment.get("PYTHONPATH") else ""
        )
        subprocess.run(
            [
                str(pepdfm_python),
                "-m",
                "domains.peptides.anticancer.round",
                "--config",
                str(config_path),
                "--output-root",
                str(output_root),
                "--asset-dir",
                str(asset_dir),
                "--active-campaign",
                str(campaign_path),
            ],
            cwd=repository_root,
            env=environment,
            check=True,
        )
    summary = read_json(summary_path)
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": config["experiment_id"],
        "round_index": round_index,
        "generator_transition": f"p{round_index}->p{round_index + 1}",
    }
    if any((summary.get(key) != value for key, value in expected.items())):
        raise ValueError("GPU transition summary identity differs")
    gate = summary.get("checkpoint_gate")
    if not isinstance(gate, Mapping) or gate.get("all_conditions_met") is not True:
        next_state = dict(state)
        next_state.update(
            phase="terminal_complete",
            terminal_status="falsified",
            terminal_reason="GPU transition failed registered non-collapse checks",
        )
        return write_state(output_root, next_state)
    artifact = summary.get("child_distribution_artifact")
    if (
        not isinstance(artifact, Mapping)
        or artifact.get("generator_id") != f"p{round_index + 1}"
    ):
        raise ValueError("GPU summary omitted the child working cohort")
    child_path = _require_hash_bound_file(artifact, name="child working cohort")
    child_id = f"p{round_index + 1}"
    manifest_path = ensure_working_distribution(
        output_root=output_root,
        distribution_id=child_id,
        source_path=child_path,
        generator_provenance={
            "kind": "all_cartesian_pair_flow_dpo_finetuned_pepdfm",
            "parent_distribution_id": state["current_distribution_id"],
            "training_goal_id": state["current_goal_id"],
            "training_summary_path": str(summary_path.resolve()),
            "training_summary_sha256": sha256_file(summary_path),
            "child_checkpoint": summary["child_checkpoint_artifact"],
        },
        length_counts=_campaign_length_counts(config),
    )
    update = {
        "round_index": round_index,
        "parent_distribution_id": state["current_distribution_id"],
        "child_distribution_id": child_id,
        "trained_goal_id": state["current_goal_id"],
        "training_summary_path": str(summary_path.resolve()),
        "training_summary_sha256": sha256_file(summary_path),
        "child_manifest_path": str(manifest_path.resolve()),
        "child_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_gate_passed": True,
    }
    immutable_json(output_root / "updates" / f"update-{round_index:02d}.json", update)
    next_state = dict(state)
    next_state.update(
        {
            "phase": "ready_to_evaluate_child",
            "current_distribution_id": child_id,
            "distribution_ids": [*state["distribution_ids"], child_id],
            "total_updates": round_index + 1,
            "updates_at_current_goal": int(state["updates_at_current_goal"]) + 1,
            "updates": [*state["updates"], update],
        }
    )
    return write_state(output_root, next_state)


def _load_source_pair_for_controller(
    *,
    roles: AgentBundle,
    output_root: Path,
    distribution_id: str,
    goal_id: str,
    goal: FrontierGoal,
    samples: Sequence[PeptideCandidate],
) -> JudgmentPair:
    return judge_pair(
        judge=roles.judge,
        goal_id=goal_id,
        judgment_key=f"accepted-{goal_id}",
        goal=goal,
        distribution_id=distribution_id,
        samples=samples,
        output_root=output_root,
    )


def _completed_semantic_tightenings(state: Mapping[str, Any]) -> int:
    return sum(
        (
            str(row.get("kind"))
            in {"newly_reachable_smallest_tightening", "exact_final_newly_reachable"}
            for row in state.get("accepted_transitions", [])
            if isinstance(row, Mapping)
        )
    )


def _uses_exact_final_primary(config: Mapping[str, Any]) -> bool:
    return (
        config.get("evaluation", {}).get("primary_metric") == EXACT_FINAL_PRIMARY_METRIC
    )


def _exact_final_newly_reachable(state: Mapping[str, Any]) -> bool:
    return any(
        (
            str(row.get("kind")) == "exact_final_newly_reachable"
            for row in state.get("accepted_transitions", [])
            if isinstance(row, Mapping)
        )
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
    count = _completed_semantic_tightenings(state)
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
    count = _completed_semantic_tightenings(state)
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


def _intermediate_goal_count(state: Mapping[str, Any]) -> int:
    return sum((str(goal_id) != FINAL_GOAL_ID for goal_id in state.get("goal_ids", [])))


def _metric_terminal_status(
    *, config: Mapping[str, Any], state: Mapping[str, Any]
) -> str:
    """Return the registered non-final terminal status.

    Legacy V13 campaigns use transition count as their primary metric.  The
    two end-to-end activity campaigns require newly reaching exact F, so every
    non-final terminal path is a primary-endpoint falsification even if useful
    intermediate transitions were completed.
    """
    if _uses_exact_final_primary(config):
        return "falsified"
    threshold = int(config["evaluation"]["success_transition_count"])
    return (
        "primary_metric_success"
        if _completed_semantic_tightenings(state) >= threshold
        else "falsified"
    )


def evaluate_child_and_choose_next(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    roles: AgentBundle,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    distribution_id = str(state["current_distribution_id"])
    goal_id = str(state["current_goal_id"])
    goal = load_goal(output_root, goal_id)
    samples = working_candidates(
        output_root=output_root,
        distribution_id=distribution_id,
        length_counts=_campaign_length_counts(config),
    )
    current_pair = _load_source_pair_for_controller(
        roles=roles,
        output_root=output_root,
        distribution_id=distribution_id,
        goal_id=goal_id,
        goal=goal,
        samples=samples,
    )
    entry_cell = load_judgment_cell(Path(str(state["current_goal_entry_cell_path"])))
    movement = choose_after_update(
        entry_cell=entry_cell,
        current_cell=current_pair.cell,
        updates_at_goal=int(state["updates_at_current_goal"]),
        minimum_enrichment=float(config["geometric_control"]["minimum_enrichment"]),
        minimum_enriched_positive_fraction=float(
            config["geometric_control"]["minimum_enriched_positive_fraction"]
        ),
        maximum_updates_per_goal=int(
            config["geometric_control"]["maximum_updates_per_goal"]
        ),
    )
    movement_path = (
        output_root / "decisions" / f"after-{distribution_id}-{goal_id}.json"
    )
    immutable_json(
        movement_path,
        {
            "schema_version": 1,
            "distribution_id": distribution_id,
            "goal_id": goal_id,
            "movement": movement.as_mapping(),
            "descriptive_95pct_stratified_bootstrap_difference": _descriptive_difference_interval(
                entry_cell, current_pair.cell, key=f"{distribution_id}-{goal_id}"
            ),
            "current_learnability": audit_learnability(current_pair.cell).as_mapping(),
            "current_categorical_report_path": str(
                (current_pair.root / "categorical_report.json").resolve()
            ),
            "no_independent_generated_validation_cohort": True,
        },
    )
    if movement.action == "stop_unproductive":
        update_limit = int(config["geometric_control"]["maximum_updates_per_goal"])
        next_state = dict(state)
        next_state.update(
            phase="terminal_complete",
            terminal_status=_metric_terminal_status(config=config, state=state),
            terminal_reason=f"{update_limit} updates did not enrich the fixed goal by ten percentage points",
        )
        return write_state(output_root, next_state)
    if goal.is_final and movement.action == "tighten":
        next_state = dict(state)
        next_state.update(
            phase="terminal_complete",
            terminal_status="reached_final_goal",
            terminal_reason="exact final goal was trainable and enriched by at least ten percentage points",
        )
        return write_state(output_root, next_state)
    update_cap_reached = int(state["total_updates"]) >= int(
        config["geometric_control"]["maximum_total_updates"]
    )
    if update_cap_reached and movement.action == "hold_and_update":
        total_limit = int(config["geometric_control"]["maximum_total_updates"])
        next_state = dict(state)
        next_state.update(
            phase="terminal_complete",
            terminal_status=_metric_terminal_status(config=config, state=state),
            terminal_reason=f"maximum {total_limit} generator updates reached while the current goal still required another update",
        )
        return write_state(output_root, next_state)
    if movement.action == "hold_and_update":
        if not audit_learnability(current_pair.cell).passed:
            next_state = dict(state)
            next_state.update(
                phase="terminal_complete",
                terminal_status=_metric_terminal_status(config=config, state=state),
                terminal_reason="held goal ceased to provide 20 stable positives and 20 stable negatives",
            )
            return write_state(output_root, next_state)
        stage_alignment_buffer(
            config=config,
            output_root=output_root,
            round_index=int(state["total_updates"]),
            distribution_id=distribution_id,
            goal_id=goal_id,
            goal=goal,
            candidates=samples,
            pair=current_pair,
            semantic_trajectory=state["goal_ids"],
        )
        next_state = dict(state)
        next_state.update(
            phase="ready_to_train",
            pending_train_cell_path=str(
                (current_pair.root / "stable_summary.json").resolve()
            ),
        )
        return write_state(output_root, next_state)
    completed_tightenings = _completed_semantic_tightenings(state)
    maximum_tightenings = int(
        config["geometric_control"]["maximum_semantic_tightenings"]
    )
    if completed_tightenings >= maximum_tightenings:
        next_state = dict(state)
        next_state.update(
            phase="terminal_complete",
            terminal_status=_metric_terminal_status(config=config, state=state),
            terminal_reason=f"maximum {maximum_tightenings} semantic transitions were already completed; no additional semantic calls or generator update is allowed",
        )
        return write_state(output_root, next_state)
    final_goal = load_goal(output_root, FINAL_GOAL_ID)
    final_current = judge_pair(
        judge=roles.judge,
        goal_id=FINAL_GOAL_ID,
        judgment_key=f"transition-{goal_id}-to-final-current",
        goal=final_goal,
        distribution_id=distribution_id,
        samples=samples,
        output_root=output_root,
    )
    entry_id = str(state["current_goal_entry_distribution_id"])
    entry_samples = working_candidates(
        output_root=output_root,
        distribution_id=entry_id,
        length_counts=_campaign_length_counts(config),
    )
    final_entry = judge_pair(
        judge=roles.judge,
        goal_id=FINAL_GOAL_ID,
        judgment_key=f"transition-{goal_id}-to-final-entry",
        goal=final_goal,
        distribution_id=entry_id,
        samples=entry_samples,
        output_root=output_root,
    )
    final_unlocked = audit_learnability(final_current.cell).passed and _at_least(
        final_current.cell.q - final_entry.cell.q,
        float(config["geometric_control"]["minimum_new_goal_unlock"]),
    )
    if final_unlocked and completed_tightenings < maximum_tightenings:
        next_state = dict(state)
        transition = {
            "kind": "exact_final_newly_reachable",
            "source_goal_id": goal_id,
            "target_goal_id": FINAL_GOAL_ID,
            "goal_entry_distribution_id": entry_id,
            "current_distribution_id": distribution_id,
            "q_entry": final_entry.cell.q,
            "q_current": final_current.cell.q,
        }
        next_state["accepted_transitions"] = [
            *state["accepted_transitions"],
            transition,
        ]
        next_state.update(
            phase="terminal_complete",
            terminal_status="reached_final_goal",
            terminal_reason="the exact immutable final goal became newly reachable; the registered protocol stops before any further fine-tuning",
        )
        return write_state(output_root, next_state)
    maximum_intermediate_goals = config["geometric_control"].get(
        "maximum_intermediate_goals"
    )
    if maximum_intermediate_goals is not None and _intermediate_goal_count(
        state
    ) >= int(maximum_intermediate_goals):
        next_state = dict(state)
        next_state.update(
            phase="terminal_complete",
            terminal_status=_metric_terminal_status(config=config, state=state),
            terminal_reason=f"G{int(maximum_intermediate_goals):02d} enriched but the exact immutable final goal was not newly reachable; G06 is forbidden",
        )
        return write_state(output_root, next_state)
    entry_source_pair = load_judgment_pair(
        Path(str(state["current_goal_entry_cell_path"])).parent
    )
    accepted = propose_reachable_goal(
        config=config,
        output_root=output_root,
        roles=roles,
        action="tighten",
        source_goal=goal,
        current_distribution_id=distribution_id,
        current_samples=samples,
        source_pair=current_pair,
        goal_ids=state["goal_ids"],
        update_count=int(state["total_updates"]),
        goal_entry_distribution_id=entry_id,
        goal_entry_samples=entry_samples,
        goal_entry_source_pair=entry_source_pair,
    )
    if accepted is None:
        next_state = dict(state)
        next_state.update(
            phase="terminal_complete",
            terminal_status=_metric_terminal_status(config=config, state=state),
            terminal_reason="no newly reachable tighter goal after one proposal and three corrections",
        )
        return write_state(output_root, next_state)
    next_goal_id = (
        f"G{len([item for item in state['goal_ids'] if item != FINAL_GOAL_ID]) + 1:02d}"
    )
    goal_path = write_goal(
        output_root=output_root,
        goal_id=next_goal_id,
        goal=accepted.proposal.goal,
        source={
            "kind": "accepted_newly_reachable_tightening",
            "proposal_path": str(accepted.proposal_path.resolve()),
            "proposal_sha256": sha256_file(accepted.proposal_path),
            "decision_path": str(accepted.decision_path.resolve()),
            "decision_sha256": sha256_file(accepted.decision_path),
        },
    )
    transition = {
        "kind": "newly_reachable_smallest_tightening",
        "source_goal_id": goal_id,
        "target_goal_id": next_goal_id,
        "goal_entry_distribution_id": entry_id,
        "current_distribution_id": distribution_id,
        "q_entry": None if accepted.entry_pair is None else accepted.entry_pair.cell.q,
        "q_current": accepted.pair.cell.q,
        "proposal_path": str(accepted.proposal_path.resolve()),
    }
    transition_count_after_acceptance = completed_tightenings + 1
    legacy_tightening_stop = (
        maximum_intermediate_goals is None
        and transition_count_after_acceptance >= maximum_tightenings
    )
    if update_cap_reached or legacy_tightening_stop:
        next_state = dict(state)
        next_state["goal_ids"] = [*state["goal_ids"], next_goal_id]
        next_state["accepted_transitions"] = [
            *state["accepted_transitions"],
            transition,
        ]
        next_state.update(
            phase="terminal_complete",
            terminal_status=_metric_terminal_status(
                config=config,
                state={
                    **next_state,
                    "accepted_transitions": next_state["accepted_transitions"],
                },
            ),
            terminal_reason="a newly reachable tighter goal was identified, after which "
            + (
                f"the maximum {maximum_tightenings} semantic tightenings had been completed"
                if legacy_tightening_stop
                else f"the maximum {int(config['geometric_control']['maximum_total_updates'])} generator updates had been completed"
            )
            + "; no further fine-tuning was allowed",
        )
        return write_state(output_root, next_state)
    return _adopt_and_stage(
        config=config,
        output_root=output_root,
        state=state,
        distribution_id=distribution_id,
        goal_id=next_goal_id,
        goal=accepted.proposal.goal,
        pair=accepted.pair,
        goal_entry_cell_path=accepted.pair.root / "stable_summary.json",
        goal_path=goal_path,
        transition=transition,
    )


def _validate_authorization(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    authorization_path: Path,
    campaign_path: Path,
) -> None:
    authorization = read_json(authorization_path)
    campaign = read_json(campaign_path)
    if authorization.get("status") != "approved" or campaign.get("status") != "active":
        raise PermissionError("V13 campaign is not approved and active")
    experiment_id = config["experiment_id"]
    if authorization.get("experiment_id") != experiment_id:
        raise PermissionError("V13 experiment identity differs")
    registration = resolve_campaign_registration(campaign, experiment_id=experiment_id)
    if authorization.get("output_root") != str(output_root.resolve()):
        raise PermissionError("authorized V13 output root differs")
    repository_root = Path(__file__).resolve().parents[3]
    template_name = str(authorization.get("config_template_file", ""))
    if not template_name:
        raise PermissionError("authorization omitted the registered config template")
    template_path = (repository_root / template_name).resolve()
    if not template_path.is_file() or authorization.get(
        "config_template_file_sha256"
    ) != sha256_file(template_path):
        raise PermissionError("authorized V13 template SHA-256 differs")
    template = read_json(template_path)
    expected = json.loads(canonical_json(template))
    expected["candidate_pool"] = {
        **dict(expected["candidate_pool"]),
        "sha256": config["candidate_pool"]["sha256"],
        "length_counts": dict(config["candidate_pool"]["length_counts"]),
    }
    expected["campaign_length_policy"] = {
        **dict(expected["campaign_length_policy"]),
        "sha256": config["campaign_length_policy"]["sha256"],
        "length_counts": dict(config["campaign_length_policy"]["length_counts"]),
        "selected_before_p0": True,
        "frozen_for_entire_campaign": True,
    }
    expected["fresh_sampling"] = {
        **dict(expected["fresh_sampling"]),
        "length_counts_per_arm": dict(
            config["fresh_sampling"]["length_counts_per_arm"]
        ),
        "frozen_evidence_reference": {
            **dict(expected["fresh_sampling"]["frozen_evidence_reference"]),
            "sha256": config["fresh_sampling"]["frozen_evidence_reference"]["sha256"],
        },
    }
    if expected != dict(config):
        raise PermissionError(
            "runtime config differs beyond registered V13 freeze bindings"
        )
    frozen_manifest_path = config_path.with_name("frozen_config_manifest.json")
    manifest = read_json(frozen_manifest_path)
    manifest_expected = {
        "template_sha256": sha256_file(template_path),
        "frozen_config_sha256": sha256_file(config_path),
        "candidate_pool_sha256": config["candidate_pool"]["sha256"],
        "frozen_evidence_reference_sha256": config["fresh_sampling"][
            "frozen_evidence_reference"
        ]["sha256"],
        "campaign_length_policy_sha256": config["campaign_length_policy"]["sha256"],
        "external_dataset_context_visible": False,
        "length_counts": dict(config["fresh_sampling"]["length_counts_per_arm"]),
    }
    if any((manifest.get(key) != value for key, value in manifest_expected.items())):
        raise PermissionError("frozen V13 config manifest differs")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repository_root, text=True
        ).strip()
    )
    if authorization.get("code_commit") != commit or dirty:
        raise PermissionError(
            "V13 real API execution requires the approved clean commit"
        )
    resources = registration.resources
    resource_expectations = {
        "config_template": template_name,
        "config_template_sha256": sha256_file(template_path),
        "derived_config_path": str(config_path.resolve()),
        "output_root": str(output_root.resolve()),
        "code_commit": commit,
    }
    for key, value in resource_expectations.items():
        if resources.get(key) != value:
            raise PermissionError(f"active campaign resource differs: {key}")
    dynamic_policy = resources.get("dynamic_length_policy")
    if not isinstance(dynamic_policy, Mapping):
        raise PermissionError("active campaign omitted V13 dynamic length policy")
    expected_policy_method = _authorized_length_policy_selection_method(config)
    if (
        _path_from_config(dynamic_policy.get("path", ""))
        != _path_from_config(config["campaign_length_policy"]["path"])
        or dynamic_policy.get("selection_method") != expected_policy_method
    ):
        raise PermissionError("active campaign V13 dynamic resource differs")
    deadline = dt.datetime.fromisoformat(
        str(authorization["deadline_utc"]).replace("Z", "+00:00")
    )
    if dt.datetime.now(dt.timezone.utc) > deadline:
        raise PermissionError("V13 authorization expired")


def _terminal_report(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    goal_rows = [
        read_json(_goal_path(output_root, goal_id)) for goal_id in state["goal_ids"]
    ]
    categorical_paths = sorted(
        output_root.glob("judgments/*/*/categorical_report.json")
    )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "protocol_version": PROTOCOL_VERSION,
        "status": state["terminal_status"],
        "reason": state["terminal_reason"],
        "immutable_final_goal": config["final_goal"],
        "goal_evolution": goal_rows,
        "accepted_transitions": state["accepted_transitions"],
        "primary_metric": _primary_metric_report(config=config, state=state),
        "generator_updates": state["updates"],
        "categorical_report_artifacts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in categorical_paths
        ],
        "external_dataset_or_reference_context_visible": False,
        "working_cohort_size": 100,
        "one_working_cohort_per_generator_state": True,
        "independent_generated_validation_cohort": False,
        "gpu_parent_diagnostic_boundary": "The GPU transition reuses the registered current 100-peptide working cohort as its parent non-collapse diagnostic; it does not sample an additional parent cohort. Only one fresh 100-peptide child is generated and becomes the next semantic working cohort.",
        "cardinal_scores_used": False,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "model_version_bindings": model_version_binding_records(output_root),
        "claim_boundary": str(config["claim_boundary"]),
    }
    secondary_metrics = _secondary_metric_reports(config=config, state=state)
    if secondary_metrics:
        report["secondary_metrics"] = secondary_metrics
    path = output_root / "final_report.json"
    immutable_json(path, report)
    return report


def _execute_until_terminal(
    *,
    config_path: Path,
    output_root: Path,
    asset_dir: Path,
    authorization_path: Path,
    campaign_path: Path,
    pepdfm_python: Path,
    dependency_overlay: Path,
) -> dict[str, Any]:
    provider_key()
    config = read_json(config_path)
    validate_config(config)
    _validate_authorization(
        config=config,
        config_path=config_path,
        output_root=output_root,
        authorization_path=authorization_path,
        campaign_path=campaign_path,
    )
    roles = build_roles(config, output_root)
    state = load_state(output_root)
    if state is None:
        state = _initialize(
            config=config, config_path=config_path, output_root=output_root
        )
    maximum_control_phases = (
        2 * int(config.get("geometric_control", {}).get("maximum_total_updates", 10))
        + 8
    )
    for _ in range(maximum_control_phases):
        phase = state["phase"]
        if phase == "need_initial_goal":
            state = discover_initial_goal(
                config=config, output_root=output_root, roles=roles, state=state
            )
        elif phase == "ready_to_train":
            state = execute_training_update(
                config=config,
                config_path=config_path,
                output_root=output_root,
                campaign_path=campaign_path,
                asset_dir=asset_dir,
                pepdfm_python=pepdfm_python,
                dependency_overlay=dependency_overlay,
                state=state,
            )
        elif phase == "ready_to_evaluate_child":
            state = evaluate_child_and_choose_next(
                config=config, output_root=output_root, roles=roles, state=state
            )
        elif phase == "terminal_complete":
            return _terminal_report(
                config=config,
                config_path=config_path,
                output_root=output_root,
                state=state,
            )
        else:
            raise ValueError(f"unknown V13 control phase: {phase}")
    raise RuntimeError("V13 control loop exceeded its bounded phase count")


def execute(
    *,
    config_path: Path,
    output_root: Path,
    asset_dir: Path,
    authorization_path: Path,
    campaign_path: Path,
    pepdfm_python: Path,
    dependency_overlay: Path,
) -> dict[str, Any]:
    try:
        return _execute_until_terminal(
            config_path=config_path,
            output_root=output_root,
            asset_dir=asset_dir,
            authorization_path=authorization_path,
            campaign_path=campaign_path,
            pepdfm_python=pepdfm_python,
            dependency_overlay=dependency_overlay,
        )
    except AllocationYieldRequired as error:
        return _write_allocation_yield(
            config_path=config_path,
            output_root=output_root,
            authorization_path=authorization_path,
            campaign_path=campaign_path,
            error=error,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--active-campaign", type=Path, required=True)
    parser.add_argument("--pepdfm-python", type=Path, required=True)
    parser.add_argument("--dependency-overlay", type=Path, required=True)
    args = parser.parse_args()
    result = execute(
        config_path=args.config.resolve(),
        output_root=args.output_root.resolve(),
        asset_dir=args.asset_dir.resolve(),
        authorization_path=args.authorization.resolve(),
        campaign_path=args.active_campaign.resolve(),
        pepdfm_python=args.pepdfm_python.absolute(),
        dependency_overlay=args.dependency_overlay.resolve(),
    )
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    if result.get("status") == "allocation_yield":
        raise SystemExit(75)


if __name__ == "__main__":
    main()
