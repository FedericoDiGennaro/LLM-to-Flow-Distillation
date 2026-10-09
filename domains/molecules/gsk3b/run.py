#!/usr/bin/env python3
"""Run the separate molecules categorical LTFD host loop."""

from __future__ import annotations

import argparse
from dataclasses import replace
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from adm.preferences.artifacts import FilePreferenceArtifactStore
from adm.preferences.clients import StructuredRequest, canonical_json, value_as_json
from domains.molecules.gsk3b.transport import GeminiUsageBudget, GeminiVertexClient
from adm.preferences.provider import bind_provider, make_structured_client, observed_provider_model, without_credentials
from domains.molecules.gsk3b.cards import FINAL_GOAL, MoleculeCandidateCard
from domains.molecules.gsk3b.teacher import (
    MoleculeGoalProposalResponse,
    MoleculeJudgmentResponse,
    MoleculeSemanticAuditResponse,
    build_audit_request,
    build_controller_request,
    build_judgment_request,
    validated_rows,
)
from domains.molecules.gsk3b.config import load_campaign_config
from domains.molecules.gsk3b.gates import at_least, judgments_learnable, require_noncollapse
from domains.molecules.gsk3b.resume import bind_configuration, control_for_round, clears_unlock
from domains.molecules.gsk3b.artifacts import (
    immutable_json,
    read_json,
    read_jsonl,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]


def _without_credentials() -> dict[str, str]:
    return without_credentials(os.environ)


def _run(command: Sequence[str], timeout: int | None = None) -> None:
    subprocess.run(
        list(command),
        cwd=ROOT,
        env=_without_credentials(),
        check=True,
        timeout=timeout,
    )


class RoleExecutor:
    def __init__(self, *, role: str, config: Mapping[str, Any], root: Path) -> None:
        role_config = config[role]
        budget = role_config["api_budget"]
        self.maximum_calls = int(role_config["maximum_provider_calls"])
        self.maximum_operational_retries = int(
            config["transport"].get("maximum_operational_retries", 2)
        )
        attempt_root = root / role / "api_attempts"
        self.calls = sum(1 for _ in attempt_root.glob("*/attempt_started.json"))
        self.store = FilePreferenceArtifactStore(root / role / "artifacts")
        bind_provider(root, legacy_glob="**/attempt_started.json")
        self.client = make_structured_client(
            native_client_class=GeminiVertexClient,
            count_tokens_on_budget_rejection=True,
            timeout=float(config["transport"]["request_timeout_seconds"]),
            attempt_root=attempt_root,
            requests_per_minute=float(
                config["transport"]["requests_per_minute_per_role"]
            ),
            usage_budget=GeminiUsageBudget(
                maximum_prompt_tokens=int(budget["maximum_prompt_tokens"]),
                maximum_output_tokens=int(budget["maximum_output_tokens"]),
                maximum_estimated_usd=float(budget["maximum_estimated_usd"]),
                input_usd_per_million_tokens=float(
                    budget["input_usd_per_million_tokens"]
                ),
                output_usd_per_million_tokens=float(
                    budget["output_usd_per_million_tokens"]
                ),
                prompt_token_overhead_per_request=int(
                    budget["prompt_token_overhead_per_request"]
                ),
            ),
        )

        # Cohort/proposal summaries can bypass execute() on resume. Restore the
        # provider revision before any new request, not only on cache hits.
        for path in sorted(attempt_root.glob("*/provider_response.json")):
            envelope = read_json(path)
            if 200 <= int(envelope["http_status"]) < 300:
                raw = envelope.get("provider_response")
                self.client.observe_model_version(observed_provider_model(raw or {}))
        for path in sorted(self.store.root.glob("*/response.json")):
            cached = self.store.load_response(path.parent.name)
            if cached is not None:
                self.client.observe_model_version_metadata(cached.raw_response)

    def execute(
        self,
        request: StructuredRequest,
        response_model: type[BaseModel],
    ) -> tuple[BaseModel, dict[str, Any]]:
        original_hash = request.request_hash
        for retry_index in range(self.maximum_operational_retries + 1):
            active_request = request
            if retry_index:
                context = request.logical_context
                context["operational_retry"] = retry_index
                active_request = replace(
                    request, logical_context_json=canonical_json(context)
                )
            self.store.save_request(active_request)
            cached = self.store.load_response(active_request.request_hash)
            if cached is not None:
                self.client.observe_model_version_metadata(cached.raw_response)
                parsed = response_model.model_validate(cached.parsed)
                usage = cached.usage
                raw = cached.raw_response
                break
            if self.calls >= self.maximum_calls:
                raise PermissionError("molecules role provider-call ceiling exhausted")
            self.calls += 1
            try:
                response = self.client.parse(active_request, response_model)
            except Exception as error:
                fatal = isinstance(error, PermissionError) or any(
                    token in str(error)
                    for token in (
                        "HTTP 400",
                        "HTTP 401",
                        "HTTP 403",
                        "HTTP 404",
                        "modelVersion changed",
                        "usage budget",
                    )
                )
                if fatal or retry_index >= self.maximum_operational_retries:
                    raise
                time.sleep(5 * (retry_index + 1))
                continue
            self.store.save_response(active_request, response)
            parsed = response.parsed
            usage = value_as_json(response.usage)
            raw = response.raw_response
            break
        else:  # pragma: no cover - bounded loop always exits or raises
            raise RuntimeError("unreachable Gemini retry state")
        raw_mapping = raw if isinstance(raw, Mapping) else {}
        return parsed, {
            "logical_request_hash": original_hash,
            "request_hash": active_request.request_hash,
            "operational_retry": retry_index,
            "usage": usage,
            "model_version": raw_mapping.get("model_version"),
        }


def verify_authorization(
    profile: Path,
    config: Mapping[str, Any],
    output_root: Path,
    authorization_path: Path,
    campaign_path: Path,
    *, expected_scopes: tuple[str, ...] = ("full_ltfd", "engineering_smoke"),
) -> None:
    authorization = read_json(authorization_path)
    campaign = read_json(campaign_path)
    if authorization.get("status") != "approved":
        raise PermissionError("molecules authorization is not approved")
    if campaign.get("status") != "active":
        raise PermissionError("molecules campaign is not active")
    if authorization.get("authorization_scope") not in expected_scopes:
        raise PermissionError("molecules authorization scope does not permit this controller")
    for key in ("experiment_id", "output_root", "config_file_sha256", "authorization_scope", "deadline_utc"):
        if not authorization.get(key) or campaign.get(key) != authorization[key]:
            raise PermissionError(f"molecules campaign/authorization {key} differs")
    if authorization.get("experiment_id") != config["experiment_id"]:
        raise PermissionError("molecules authorization experiment differs")
    if authorization.get("config_file_sha256") != sha256_file(profile):
        raise PermissionError("molecules authorized profile changed")
    resolved_hash = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    if authorization.get("resolved_config_sha256") != resolved_hash:
        raise PermissionError("molecules authorized base/resolved config changed")
    if Path(str(authorization.get("output_root", ""))).resolve() != output_root:
        raise PermissionError("molecules authorized output root differs")
    deadline = dt.datetime.fromisoformat(
        str(authorization["deadline_utc"]).replace("Z", "+00:00")
    )
    if deadline <= dt.datetime.now(dt.timezone.utc):
        raise PermissionError("molecules authorization expired")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if authorization.get("code_commit") != commit or dirty:
        raise PermissionError("molecules execution requires the authorized clean code commit")


def sample_generator_command(
    *,
    python: Path,
    checkpoint: Path,
    n_atoms_hist: Path,
    marginal_dists: Path,
    generation: Path,
    count: int,
    batch_size: int,
    seed: int,
    n_timesteps: int,
    priors_from: Path | None = None,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "domains.molecules.gsk3b.sample",
        "--checkpoint",
        str(checkpoint),
        "--n-atoms-hist",
        str(n_atoms_hist),
        "--marginal-dists",
        str(marginal_dists),
        "--output-dir",
        str(generation),
        "--count",
        str(count),
        "--batch-size",
        str(min(batch_size, count)),
        "--seed",
        str(seed),
        "--n-timesteps",
        str(n_timesteps),
    ]
    if priors_from is not None:
        command.extend(["--priors-from", str(priors_from)])
    return command


def generate_cohort(
    *,
    checkpoint: Path,
    root: Path,
    count: int,
    batch_size: int,
    seed: int,
    n_timesteps: int,
    role: str,
    config_path: Path,
    python: Path,
    n_atoms_hist: Path,
    marginal_dists: Path,
    priors_from: Path | None = None,
    maximum_stage_attempts: int = 3,
) -> dict[str, Any]:
    receipt_path = root / "cohort_receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        generation = Path(receipt["generation"])
        pool = Path(receipt["pool"])
        if not (generation / "samples.pt").is_file() or not (
            pool / "pool_manifest.json"
        ).is_file():
            raise RuntimeError("completed cohort receipt points to missing artifacts")
    else:
        attempts_root = root / "attempts"
        used = sorted(attempts_root.glob("attempt-*"))
        attempt = None
        generation = None
        pool = None
        run_generation = True
        run_pool = True
        if used:
            latest = used[-1]
            retained_generation = latest / "generation"
            if (retained_generation / "samples.pt").is_file():
                # Sampling is expensive and deterministic. If a later
                # qualification/dependency step failed, bind the retry to the
                # retained sample artifact instead of drawing it again.
                attempt = latest
                generation = retained_generation
                existing_pools = [
                    path
                    for path in [latest / "pool", *sorted(latest.glob("pool-recovery-*"))]
                    if path.exists()
                ]
                completed_pools = [
                    path
                    for path in existing_pools
                    if (path / "pool_manifest.json").is_file()
                    and (path / "candidate_pool.jsonl").is_file()
                ]
                if completed_pools:
                    pool = completed_pools[-1]
                    run_pool = False
                else:
                    if len(existing_pools) >= maximum_stage_attempts:
                        raise RuntimeError(
                            "cohort exhausted preserved pool recoveries for "
                            f"retained generation: {latest}"
                        )
                    pool = (
                        latest / "pool"
                        if not (latest / "pool").exists()
                        else latest / f"pool-recovery-{len(existing_pools):02d}"
                    )
                run_generation = False
        if attempt is None:
            if len(used) >= maximum_stage_attempts:
                raise RuntimeError(
                    f"cohort exhausted {maximum_stage_attempts} preserved stage attempts: {root}"
                )
            attempt = attempts_root / f"attempt-{len(used) + 1:02d}"
            # Materialize the attempt before launching so even an immediate
            # child failure is durable and cannot be overwritten on re-entry.
            attempt.mkdir(parents=True, exist_ok=False)
            generation = attempt / "generation"
            pool = attempt / "pool"
        assert generation is not None and pool is not None
        if run_generation:
            _run(
                sample_generator_command(
                    python=python,
                    checkpoint=checkpoint,
                    n_atoms_hist=n_atoms_hist,
                    marginal_dists=marginal_dists,
                    generation=generation,
                    count=count,
                    batch_size=batch_size,
                    seed=seed,
                    n_timesteps=n_timesteps,
                    priors_from=priors_from,
                ),
                timeout=7200,
            )
        if run_pool:
            _run(
                [
                    str(python),
                    "-m",
                    "domains.molecules.gsk3b.pool",
                    "--samples",
                    str(generation / "samples.pt"),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(pool),
                    "--python",
                    str(python),
                    "--role",
                    role,
                ],
                timeout=max(3600, 120 * count),
            )
        immutable_json(
            receipt_path,
            {
                "schema_version": 1,
                "attempt": attempt.name,
                "checkpoint": str(checkpoint),
                "seed": seed,
                "generation": str(generation.resolve()),
                "pool": str(pool.resolve()),
                "samples_sha256": sha256_file(generation / "samples.pt"),
                "pool_manifest_sha256": sha256_file(pool / "pool_manifest.json"),
            },
        )
    return {
        "root": root,
        "generation": generation,
        "pool": pool,
        "manifest": read_json(pool / "pool_manifest.json"),
        "rows": read_jsonl(pool / "candidate_pool.jsonl"),
    }


def judge_cohort(
    cohort: Mapping[str, Any],
    *,
    goal: str,
    goal_id: str,
    iteration: int,
    config: Mapping[str, Any],
    executor: RoleExecutor,
    output_root: Path,
) -> dict[str, Any]:
    completed = output_root / goal_id / "stable_summary.json"
    if completed.is_file():
        return read_json(completed)
    rows = list(cohort["rows"])
    cards = {
        row["item_id"]: MoleculeCandidateCard.model_validate(row["card"])
        for row in rows
    }
    repeat_results: list[dict[str, Any]] = []
    judge_config = config["candidate_judge"]
    for repeat_index in (1, 2):
        order = sorted(cards)
        random.Random(
            int(judge_config["provider_order_seed"])
            + iteration * 100
            + repeat_index
        ).shuffle(order)
        combined: dict[str, dict[str, Any]] = {}
        metadata = []
        chunk_size = int(judge_config["chunk_size"])
        for offset in range(0, len(order), chunk_size):
            chunk = [cards[item] for item in order[offset : offset + chunk_size]]
            request = build_judgment_request(
                chunk,
                current_goal=goal,
                model=str(judge_config["model"]),
                max_output_tokens=int(judge_config["max_output_tokens"]),
                campaign_id=str(config["experiment_id"]),
                iteration=iteration,
                repeat_index=repeat_index,
            )
            parsed, request_metadata = executor.execute(
                request, MoleculeJudgmentResponse
            )
            for result in validated_rows(parsed, chunk):
                combined[result.item_id] = result.model_dump(mode="json")
            metadata.append(request_metadata)
        if set(combined) != set(cards):
            raise ValueError("molecule judgment repeat did not cover all cards")
        record = {
            "repeat_index": repeat_index,
            "results": [combined[item] for item in sorted(combined)],
            "requests": metadata,
        }
        immutable_json(output_root / goal_id / f"repeat-{repeat_index}.json", record)
        repeat_results.append(record)

    first = {row["item_id"]: row for row in repeat_results[0]["results"]}
    second = {row["item_id"]: row for row in repeat_results[1]["results"]}
    stable = {
        item: (
            first[item]["final_label"]
            if first[item]["final_label"] == second[item]["final_label"]
            else "unstable"
        )
        for item in sorted(cards)
    }
    counts = {
        label: sum(value == label for value in stable.values())
        for label in ("positive", "negative", "abstain", "unstable")
    }
    agreement = sum(value != "unstable" for value in stable.values()) / len(stable)
    abstain_fractions = [
        sum(row["final_label"] == "abstain" for row in repeat["results"]) / len(cards)
        for repeat in repeat_results
    ]
    attempts = int(cohort["manifest"]["generation_attempt_count"])
    result = {
        "schema_version": 1,
        "artifact_type": "molecules_two_pass_categorical_judgment",
        "goal": goal,
        "goal_id": goal_id,
        "generation_attempt_count": attempts,
        "judgeable_count": len(cards),
        "stable_labels": stable,
        "repeat_artifacts": [
            str((output_root / goal_id / f"repeat-{index}.json").resolve())
            for index in (1, 2)
        ],
        "counts": counts,
        "exact_agreement": agreement,
        "abstain_fraction_by_pass": abstain_fractions,
        "q_positive_over_all_attempts": counts["positive"] / attempts,
        "all_generation_attempts_retained_in_denominator": True,
        "gnina_visible": False,
    }
    immutable_json(output_root / goal_id / "stable_summary.json", result)
    return result


def two_pass_feedback(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Expose exact categorical passes and scientific rationales to the controller."""

    paths = result.get("repeat_artifacts")
    if not isinstance(paths, list) or len(paths) != 2:
        raise ValueError("stable judgment is missing its two repeat artifacts")
    repeats = [read_json(Path(str(path))) for path in paths]
    indexed = [
        {row["item_id"]: row for row in repeat["results"]} for repeat in repeats
    ]
    labels = result["stable_labels"]
    if set(indexed[0]) != set(labels) or set(indexed[1]) != set(labels):
        raise ValueError("two-pass rationale coverage differs from stable labels")
    return {
        item_id: {
            "pass_1_label": indexed[0][item_id]["final_label"],
            "pass_1_rationale": indexed[0][item_id]["rationale"],
            "pass_1_critical_unknowns": indexed[0][item_id]["critical_unknowns"],
            "pass_2_label": indexed[1][item_id]["final_label"],
            "pass_2_rationale": indexed[1][item_id]["rationale"],
            "pass_2_critical_unknowns": indexed[1][item_id]["critical_unknowns"],
            "stable_label": labels[item_id],
        }
        for item_id in sorted(labels)
    }


def representative_failure_feedback(
    result: Mapping[str, Any], *, maximum_per_label: int = 2
) -> dict[str, Any]:
    feedback = two_pass_feedback(result)
    representatives: list[dict[str, Any]] = []
    for label in ("positive", "negative", "abstain", "unstable"):
        members = [
            {"item_id": item_id, **row}
            for item_id, row in feedback.items()
            if row["stable_label"] == label
        ]
        representatives.extend(members[:maximum_per_label])
    return {
        "counts": dict(result["counts"]),
        "exact_agreement": result["exact_agreement"],
        "abstain_fraction_by_pass": list(result["abstain_fraction_by_pass"]),
        "representative_both_pass_outcomes": representatives,
    }


def learnable(result: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    return judgments_learnable(result, config["curriculum_discovery"])


def propose_goal(
    cohort: Mapping[str, Any],
    source_result: Mapping[str, Any],
    *,
    final_result: Mapping[str, Any],
    source_adoption_cohort: Mapping[str, Any] | None,
    source_adoption_result: Mapping[str, Any] | None,
    source_goal: str,
    requested_action: str,
    iteration: int,
    config: Mapping[str, Any],
    controller: RoleExecutor,
    auditor: RoleExecutor,
    output_root: Path,
    previous_failure: Mapping[str, Any] | None = None,
    attempt: int = 1,
) -> tuple[MoleculeGoalProposalResponse, MoleculeSemanticAuditResponse, bool]:
    cards = [
        MoleculeCandidateCard.model_validate(row["card"]) for row in cohort["rows"]
    ]
    controller_config = config["curriculum_controller"]
    adoption_rows: list[dict[str, Any]] = []
    if source_adoption_cohort is not None and source_adoption_result is not None:
        adoption_feedback = two_pass_feedback(source_adoption_result)
        adoption_rows = [
            {
                "card": row["card"],
                "source_goal_two_pass_feedback": adoption_feedback[row["item_id"]],
            }
            for row in source_adoption_cohort["rows"]
        ]
    request = build_controller_request(
        cards,
        two_pass_feedback(source_result),
        final_goal_feedback=two_pass_feedback(final_result),
        source_goal_adoption_cohort=adoption_rows,
        source_goal=source_goal,
        requested_action=requested_action,  # type: ignore[arg-type]
        model=str(controller_config["model"]),
        max_output_tokens=int(controller_config["max_output_tokens"]),
        campaign_id=str(config["experiment_id"]),
        iteration=iteration,
        proposal_attempt=attempt,
        previous_failure=previous_failure,
    )
    proposal, proposal_metadata = controller.execute(
        request, MoleculeGoalProposalResponse
    )
    auditor_config = config["independent_auditor"]
    audit_request = build_audit_request(
        proposal,
        source_goal=source_goal,
        requested_action=requested_action,  # type: ignore[arg-type]
        model=str(auditor_config["model"]),
        max_output_tokens=int(auditor_config["max_output_tokens"]),
        campaign_id=str(config["experiment_id"]),
        iteration=iteration,
        proposal_attempt=attempt,
    )
    audit, audit_metadata = auditor.execute(
        audit_request, MoleculeSemanticAuditResponse
    )
    accepted = (
        proposal.action == requested_action
        and proposal.assessable_from_registered_cards
        and not proposal.numerical_descriptor_cutoff_used
        and not proposal.hidden_evidence_used
        and not proposal.receptor_binding_success_claimed
        and audit.decision == "accept"
        and audit.requested_action_followed
        and audit.semantic_direction_valid
        and audit.assessable_from_registered_cards
        and audit.not_numeric_cutoff_or_synonym
        and audit.hidden_evidence_excluded
        and audit.structural_proxy_limits_explicit
        and audit.scientific_success_not_claimed
    )
    immutable_json(
        output_root / f"proposal-{attempt}.json",
        {
            "proposal": proposal.model_dump(mode="json"),
            "proposal_metadata": proposal_metadata,
            "audit": audit.model_dump(mode="json"),
            "audit_metadata": audit_metadata,
            "accepted": accepted,
        },
    )
    return proposal, audit, accepted


def existing_goal_proposal(
    output_root: Path, attempt: int
) -> tuple[MoleculeGoalProposalResponse, MoleculeSemanticAuditResponse, bool] | None:
    path = output_root / f"proposal-{attempt}.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    return (
        MoleculeGoalProposalResponse.model_validate(payload["proposal"]),
        MoleculeSemanticAuditResponse.model_validate(payload["audit"]),
        bool(payload["accepted"]),
    )


def run_dpo_update(
    *,
    root: Path,
    stage: str,
    cohort: Mapping[str, Any],
    stable_result: Mapping[str, Any],
    parent_checkpoint: Path,
    profile: Path,
    flowmol_python: Path,
    n_atoms_hist: Path,
    marginal_dists: Path,
    maximum_stage_attempts: int,
) -> tuple[Path, Path]:
    """Run or resume one update without overwriting a failed attempt."""

    receipt_path = root / "update_receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        result_path = Path(receipt["training_result"])
        child = Path(receipt["child_checkpoint"])
        if not result_path.is_file() or not child.is_file():
            raise RuntimeError("update receipt points to missing artifacts")
        return child, result_path
    attempts_root = root / "attempts"
    used = sorted(attempts_root.glob("attempt-*"))
    if len(used) >= maximum_stage_attempts:
        raise RuntimeError(
            f"DPO update exhausted {maximum_stage_attempts} preserved attempts: {root}"
        )
    attempt = attempts_root / f"attempt-{len(used) + 1:02d}"
    attempt.mkdir(parents=True, exist_ok=False)
    labels_path = Path(str(stable_result["repeat_artifacts"][0])).parent / (
        "stable_summary.json"
    )
    _run(
        [
            str(flowmol_python),
            "-m",
            "domains.molecules.gsk3b.round",
            "--pool",
            str(cohort["pool"] / "candidate_pool.jsonl"),
            "--stable-labels",
            str(labels_path),
            "--parent-checkpoint",
            str(parent_checkpoint),
            "--n-atoms-hist",
            str(n_atoms_hist),
            "--marginal-dists",
            str(marginal_dists),
            "--config",
            str(profile),
            "--stage",
            stage,
            "--output-dir",
            str(attempt),
        ],
        timeout=7200,
    )
    child = attempt / "last.ckpt"
    result_path = attempt / "training_result.json"
    if not child.is_file() or not result_path.is_file():
        raise RuntimeError("FlowMol DPO process exited without complete artifacts")
    immutable_json(
        receipt_path,
        {
            "schema_version": 1,
            "stage": stage,
            "attempt": attempt.name,
            "parent_checkpoint": str(parent_checkpoint),
            "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
            "child_checkpoint": str(child.resolve()),
            "child_checkpoint_sha256": sha256_file(child),
            "training_result": str(result_path.resolve()),
            "training_result_sha256": sha256_file(result_path),
        },
    )
    return child, result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--flowmol-python", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--active-campaign", type=Path, required=True)
    args = parser.parse_args()
    profile = args.config.resolve()
    config = load_campaign_config(profile)
    if config.get("preference_source") == "tdc_gsk3b_only":
        raise ValueError("use the baseline controller for a reward-only profile")
    if str(config["final_goal"]) != FINAL_GOAL:
        raise ValueError(
            "configured final goal differs from the drug prompt/card goal; "
            "update the configuration, prompts, cards and validation rules together"
        )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    verify_authorization(
        profile,
        config,
        output_root,
        args.authorization.resolve(),
        args.active_campaign.resolve(),
    )
    original_config, effective_frozen_config = bind_configuration(config, output_root, create=True)
    frozen_config = output_root / "frozen_config.json"

    api_root = output_root / "api"
    judge = RoleExecutor(role="candidate_judge", config=config, root=api_root)
    controller = RoleExecutor(
        role="curriculum_controller", config=config, root=api_root
    )
    auditor = RoleExecutor(
        role="independent_auditor", config=config, root=api_root
    )
    checkpoint = args.checkpoint.resolve()
    n_timesteps = int(config["fresh_sampling"]["flow_timesteps"])
    p0 = generate_cohort(
        checkpoint=checkpoint,
        root=output_root / "p0",
        count=int(config["candidate_pool"]["candidate_count"]),
        batch_size=int(config["fresh_sampling"]["batch_size"]),
        seed=int(config["candidate_pool"]["seed"]),
        n_timesteps=n_timesteps,
        role="training",
        config_path=frozen_config,
        python=args.flowmol_python.absolute(),
        n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
    )
    final_result = judge_cohort(
        p0,
        goal=str(config["final_goal"]),
        goal_id="F-p0",
        iteration=0,
        config=config,
        executor=judge,
        output_root=output_root / "judgments",
    )

    training_cohort = p0
    source_goal = str(config["final_goal"])
    source_result = final_result
    working_final_result = final_result
    frozen_p0_samples = p0["generation"] / "samples.pt"
    rounds: list[dict[str, Any]] = []
    stop_reason = "maximum_rounds"
    held_selection: dict[str, Any] | None = None
    updates_on_current_goal = 0
    current_goal_adoption_cohort: Mapping[str, Any] | None = None
    current_goal_adoption_result: Mapping[str, Any] | None = None
    model = checkpoint
    goal_index = 0
    working_f_streak = 0
    control = config["geometric_control"]
    heuristic = config["heuristic_control"]
    recovery = config.get("recovery", {})
    maximum_stage_attempts = int(recovery.get("maximum_stage_attempts", 3))
    max_rounds = int(config["curriculum_loop"]["maximum_alignment_rounds"])
    max_total_updates = int(control["maximum_total_updates"])
    max_updates_per_goal = int(control["maximum_updates_per_goal"])
    max_tightenings = int(control["maximum_semantic_tightenings"])
    for round_index in range(1, max_rounds + 1):
        control = control_for_round(config, original_config, round_index)
        frozen_config = (
            effective_frozen_config if control is config["geometric_control"]
            else output_root / "frozen_config.json"
        )
        if len(rounds) >= max_total_updates:
            stop_reason = "maximum_total_updates"
            break
        round_root = output_root / "rounds" / f"round-{round_index:02d}"
        selected_goal = held_selection
        held_selection = None
        held_retry = selected_goal is not None
        requested_action = "broaden" if goal_index == 0 else "tighten"
        if selected_goal is None:
            if requested_action == "tighten" and goal_index - 1 >= max_tightenings:
                stop_reason = "maximum_semantic_tightenings"
                break
            previous_failure = None
            for attempt in range(1, int(control["maximum_proposal_attempts"]) + 1):
                restored = existing_goal_proposal(round_root / "goal_selection", attempt)
                if restored is not None:
                    proposal, _audit, accepted = restored
                else:
                    proposal, _audit, accepted = propose_goal(
                        training_cohort,
                        source_result,
                        final_result=working_final_result,
                        source_adoption_cohort=current_goal_adoption_cohort,
                        source_adoption_result=current_goal_adoption_result,
                        source_goal=source_goal,
                        requested_action=requested_action,
                        iteration=round_index,
                        config=config,
                        controller=controller,
                        auditor=auditor,
                        output_root=round_root / "goal_selection",
                        previous_failure=previous_failure,
                        attempt=attempt,
                    )
                if not accepted:
                    previous_failure = {
                        "reason": "semantic_audit_rejected",
                        "proposal": proposal.model_dump(mode="json"),
                    }
                    continue
                tentative_goal_id = f"G{goal_index + 1:02d}"
                goal_result = judge_cohort(
                    training_cohort,
                    goal=proposal.proposed_goal,
                    goal_id=f"{tentative_goal_id}-entry-attempt-{attempt}",
                    iteration=round_index,
                    config=config,
                    executor=judge,
                    output_root=output_root / "judgments",
                )
                if not learnable(goal_result, config):
                    previous_failure = {
                        "reason": "goal_not_learnable",
                        "previous_proposal": proposal.model_dump(mode="json"),
                        **representative_failure_feedback(goal_result),
                    }
                    continue
                if requested_action == "broaden":
                    unlock = (
                        goal_result["q_positive_over_all_attempts"]
                        - final_result["q_positive_over_all_attempts"]
                    )
                    unlock_reference = final_result["goal_id"]
                    required_unlock = float(control["minimum_initial_broadening_gain"])
                else:
                    if current_goal_adoption_cohort is None:
                        raise RuntimeError("tightening lacks source-goal adoption cohort")
                    adoption_result = judge_cohort(
                        current_goal_adoption_cohort,
                        goal=proposal.proposed_goal,
                        goal_id=f"{tentative_goal_id}-adoption-baseline-attempt-{attempt}",
                        iteration=round_index,
                        config=config,
                        executor=judge,
                        output_root=output_root / "judgments",
                    )
                    unlock = (
                        goal_result["q_positive_over_all_attempts"]
                        - adoption_result["q_positive_over_all_attempts"]
                    )
                    unlock_reference = adoption_result["goal_id"]
                    required_unlock = float(control["minimum_new_goal_unlock"])
                if not clears_unlock(unlock, required_unlock, config, round_index):
                    previous_failure = {
                        "reason": "goal_not_newly_reachable",
                        "previous_proposal": proposal.model_dump(mode="json"),
                        "q_current": goal_result["q_positive_over_all_attempts"],
                        "q_reference": (
                            goal_result["q_positive_over_all_attempts"] - unlock
                        ),
                        "observed_unlock": unlock,
                        "required_unlock": required_unlock,
                        **representative_failure_feedback(goal_result),
                    }
                    continue
                selected_goal = {
                    "goal": proposal.proposed_goal,
                    "goal_id": tentative_goal_id,
                    "training_result": goal_result,
                    "adoption_q": goal_result["q_positive_over_all_attempts"],
                    "unlock": unlock,
                    "unlock_reference": unlock_reference,
                    "goal_action": requested_action,
                }
                goal_index += 1
                current_goal_adoption_cohort = training_cohort
                current_goal_adoption_result = goal_result
                updates_on_current_goal = 0
                break
        if selected_goal is None:
            stop_reason = "no_audited_learnable_newly_reachable_goal"
            break
        goal = str(selected_goal["goal"])
        goal_id = str(selected_goal["goal_id"])
        entry_result = selected_goal["training_result"]
        if not learnable(entry_result, config):
            stop_reason = "held_goal_no_longer_learnable"
            break
        updates_on_current_goal += 1
        child_model, training_result_path = run_dpo_update(
            root=round_root / "update",
            stage="curriculum",
            cohort=training_cohort,
            stable_result=entry_result,
            parent_checkpoint=model,
            profile=profile,
            flowmol_python=args.flowmol_python.absolute(),
            n_atoms_hist=args.n_atoms_hist.resolve(),
            marginal_dists=args.marginal_dists.resolve(),
            maximum_stage_attempts=maximum_stage_attempts,
        )
        seed = int(config["fresh_sampling"]["round_1_seeds"][0]) + (
            round_index - 1
        ) * int(config["curriculum_loop"]["round_seed_stride"])
        count = int(config["fresh_sampling"]["num_attempts_per_arm"])
        child_development = generate_cohort(
            checkpoint=child_model,
            root=round_root / "development/child",
            count=count,
            batch_size=int(config["fresh_sampling"]["batch_size"]),
            seed=seed,
            n_timesteps=n_timesteps,
            role="development",
            config_path=frozen_config,
            python=args.flowmol_python.absolute(),
            n_atoms_hist=args.n_atoms_hist.resolve(),
            marginal_dists=args.marginal_dists.resolve(),
            priors_from=training_cohort["generation"] / "samples.pt",
            maximum_stage_attempts=maximum_stage_attempts,
        )
        noncollapse_path = round_root / "noncollapse.json"
        if not noncollapse_path.is_file():
            _run(
                [
                    str(args.flowmol_python.absolute()),
                    "-m",
                    "domains.molecules.gsk3b.validate_samples",
                    "--parent-samples",
                    str(training_cohort["generation"] / "samples.pt"),
                    "--frozen-p0-samples",
                    str(frozen_p0_samples),
                    "--child-samples",
                    str(child_development["generation"] / "samples.pt"),
                    "--parent-checkpoint",
                    str(model),
                    "--child-checkpoint",
                    str(child_model),
                    "--config",
                    str(frozen_config),
                    "--output",
                    str(noncollapse_path),
                ]
            )
        require_noncollapse(noncollapse_path, inputs={
            "parent_checkpoint": model, "child_checkpoint": child_model,
            "parent_samples": training_cohort["generation"] / "samples.pt",
            "frozen_p0_samples": frozen_p0_samples,
            "child_samples": child_development["generation"] / "samples.pt",
        })
        child_result = judge_cohort(
            child_development,
            goal=goal,
            goal_id=f"{goal_id}-p{round_index}",
            iteration=round_index,
            config=config,
            executor=judge,
            output_root=output_root / "judgments",
        )
        child_final_result = judge_cohort(
            child_development,
            goal=str(config["final_goal"]),
            goal_id=f"F-p{round_index}",
            iteration=round_index,
            config=config,
            executor=judge,
            output_root=output_root / "judgments",
        )
        movement = {
            "adoption_q": selected_goal["adoption_q"],
            "parent_working_q": entry_result["q_positive_over_all_attempts"],
            "child_q": child_result["q_positive_over_all_attempts"],
        }
        movement["child_minus_adoption"] = (
            movement["child_q"] - movement["adoption_q"]
        )
        movement["child_minus_parent_working"] = (
            movement["child_q"] - movement["parent_working_q"]
        )
        enriched = (
            at_least(movement["child_q"], float(control["minimum_enriched_positive_fraction"]))
            and at_least(movement["child_minus_adoption"], float(control["minimum_enrichment"]))
        )
        f_q = child_final_result["q_positive_over_all_attempts"]
        f_reachable = (
            at_least(f_q, float(heuristic["working_F_floor"]))
            and at_least(f_q - final_result["q_positive_over_all_attempts"],
                         float(heuristic["working_F_gain_over_p0"]))
        )
        working_f_streak = working_f_streak + 1 if f_reachable else 0
        round_record = {
            "round": round_index,
            "goal_id": goal_id,
            "goal": goal,
            "goal_action": "hold" if held_retry else selected_goal["goal_action"],
            "entry_judgment": entry_result["goal_id"],
            "new_goal_unlock": selected_goal["unlock"],
            "new_goal_unlock_reference": selected_goal["unlock_reference"],
            "child_development_judgment": child_result["goal_id"],
            "final_goal_monitor_judgment": child_final_result["goal_id"],
            "final_goal_q": f_q,
            "final_goal_reachability_streak": working_f_streak,
            "movement": movement,
            "enriched": enriched,
            "model": str(child_model),
            "training_result": str(training_result_path),
            "noncollapse": str(noncollapse_path),
        }
        immutable_json(round_root / "round_result.json", round_record)
        rounds.append(round_record)
        model = child_model
        training_cohort = child_development
        source_goal = goal
        source_result = child_result
        working_final_result = child_final_result
        if (
            bool(heuristic["working_F_stop_enabled"])
            and working_f_streak >= int(heuristic["working_F_consecutive_cohorts"])
        ):
            stop_reason = "working_final_goal_reachable"
            break
        if not enriched:
            if (
                updates_on_current_goal < max_updates_per_goal
                and round_index < max_rounds
                and len(rounds) < max_total_updates
            ):
                held_selection = {
                    **selected_goal,
                    "training_result": child_result,
                    "goal_action": "hold",
                }
                stop_reason = "held_goal_for_another_update"
                continue
            stop_reason = "goal_not_enriched_after_bounded_updates"
            break
        updates_on_current_goal = 0
        if round_index == max_rounds:
            stop_reason = "maximum_rounds"
        elif len(rounds) >= max_total_updates:
            stop_reason = "maximum_total_updates"

    final_f_rounds: list[dict[str, Any]] = []
    final_control = config["final_f_continuation"]
    final_entry_eligible = (
        (
            not bool(final_control.get("requires_at_least_one_curriculum_update"))
            or bool(rounds)
        )
        and judgments_learnable(working_final_result, config["curriculum_discovery"],
                                count_limits=final_control)
    )
    if bool(final_control["enabled"]) and final_entry_eligible:
        for final_index in range(1, int(final_control["maximum_updates"]) + 1):
            if not judgments_learnable(working_final_result, config["curriculum_discovery"],
                                       count_limits=final_control):
                break
            final_root = output_root / "final_f" / f"round-{final_index:02d}"
            child_model, training_result_path = run_dpo_update(
                root=final_root / "update",
                stage="final_f",
                cohort=training_cohort,
                stable_result=working_final_result,
                parent_checkpoint=model,
                profile=profile,
                flowmol_python=args.flowmol_python.absolute(),
                n_atoms_hist=args.n_atoms_hist.resolve(),
                marginal_dists=args.marginal_dists.resolve(),
                maximum_stage_attempts=maximum_stage_attempts,
            )
            seed = int(final_control["sampling_seed"]) + final_index - 1
            child = generate_cohort(
                checkpoint=child_model,
                root=final_root / "development/child",
                count=int(config["fresh_sampling"]["num_attempts_per_arm"]),
                batch_size=int(config["fresh_sampling"]["batch_size"]),
                seed=seed,
                n_timesteps=n_timesteps,
                role="development",
                config_path=frozen_config,
                python=args.flowmol_python.absolute(),
                n_atoms_hist=args.n_atoms_hist.resolve(),
                marginal_dists=args.marginal_dists.resolve(),
                priors_from=training_cohort["generation"] / "samples.pt",
                maximum_stage_attempts=maximum_stage_attempts,
            )
            noncollapse_path = final_root / "noncollapse.json"
            if not noncollapse_path.is_file():
                _run(
                    [
                        str(args.flowmol_python.absolute()),
                        "-m",
                        "domains.molecules.gsk3b.validate_samples",
                        "--parent-samples",
                        str(training_cohort["generation"] / "samples.pt"),
                        "--frozen-p0-samples",
                        str(frozen_p0_samples),
                        "--child-samples",
                        str(child["generation"] / "samples.pt"),
                        "--parent-checkpoint",
                        str(model),
                        "--child-checkpoint",
                        str(child_model),
                        "--config",
                        str(frozen_config),
                        "--output",
                        str(noncollapse_path),
                    ]
                )
            require_noncollapse(noncollapse_path, inputs={
                "parent_checkpoint": model, "child_checkpoint": child_model,
                "parent_samples": training_cohort["generation"] / "samples.pt",
                "frozen_p0_samples": frozen_p0_samples,
                "child_samples": child["generation"] / "samples.pt",
            })
            final_result_after_update = judge_cohort(
                child,
                goal=str(config["final_goal"]),
                goal_id=f"F-final-p{final_index}",
                iteration=max_rounds + final_index,
                config=config,
                executor=judge,
                output_root=output_root / "judgments",
            )
            record = {
                "round": final_index,
                "trained_goal_id": "F",
                "child_final_goal_judgment": final_result_after_update["goal_id"],
                "entry_q": working_final_result["q_positive_over_all_attempts"],
                "child_q": final_result_after_update["q_positive_over_all_attempts"],
                "entry_counts": dict(working_final_result["counts"]),
                "child_counts": dict(final_result_after_update["counts"]),
                "model": str(child_model),
                "training_result": str(training_result_path),
                "noncollapse": str(noncollapse_path),
            }
            immutable_json(final_root / "round_result.json", record)
            final_f_rounds.append(record)
            model = child_model
            training_cohort = child
            working_final_result = final_result_after_update

    result = {
        "schema_version": 1,
        "artifact_type": "drugs_ltfd_flowmol3_v1",
        "experiment_id": config["experiment_id"],
        "completed_round_count": len(rounds),
        "rounds": rounds,
        "curriculum_stop_reason": stop_reason,
        "p0_final_goal_q": final_result["q_positive_over_all_attempts"],
        "curriculum_terminal_final_goal_q": (
            rounds[-1]["final_goal_q"] if rounds else final_result["q_positive_over_all_attempts"]
        ),
        "final_f_entry_eligible": final_entry_eligible,
        "final_f_updates": final_f_rounds,
        "final_f_update_count": len(final_f_rounds),
        "terminal_working_final_goal_q": working_final_result[
            "q_positive_over_all_attempts"
        ],
        "final_model": str(model),
        "provider_calls": {
            "candidate_judge": judge.calls,
            "curriculum_controller": controller.calls,
            "independent_auditor": auditor.calls,
        },
        "gnina_used": False,
        "terminal_validation_required": True,
        "scientific_success_claimed": False,
    }
    immutable_json(output_root / "campaign_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not rounds and not final_f_rounds:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
