#!/usr/bin/env python3
"""Independently validate the complete drug LTFD artifact chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

from domains.molecules.drd2.config import load_campaign_config
from domains.molecules.drd2.gates import judgments_learnable, noncollapse_valid
from domains.molecules.drd2.resume import bind_configuration, control_for_round, clears_unlock, goal_passed
from domains.molecules.drd2.artifacts import immutable_json, read_json


_FULL_CARTESIAN = "full_cartesian_stable_positive_x_stable_negative"
_CAPPED_CARTESIAN = f"{_FULL_CARTESIAN}_capped"


def _close(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _validate_judgment(
    root: Path,
    judgment_id: str,
    *,
    expected_goal: str,
) -> dict[str, Any]:
    judgment = read_json(root / "judgments" / judgment_id / "stable_summary.json")
    if judgment.get("artifact_type") != "molecules_two_pass_categorical_judgment":
        raise ValueError("molecules judgment has the wrong artifact type")
    if judgment.get("goal_id") != judgment_id or judgment.get("goal") != expected_goal:
        raise ValueError("molecules judgment goal binding differs")
    if (
        judgment.get("gnina_visible") is not False
        or judgment.get("all_generation_attempts_retained_in_denominator") is not True
    ):
        raise ValueError("molecules judgment violated its evidence boundary")
    counts = judgment.get("counts")
    expected_labels = {"positive", "negative", "abstain", "unstable"}
    if not isinstance(counts, dict) or set(counts) != expected_labels:
        raise ValueError("molecules judgment count schema differs")
    attempts = int(judgment["generation_attempt_count"])
    judgeable = int(judgment["judgeable_count"])
    if (
        attempts <= 0
        or judgeable <= 0
        or judgeable > attempts
        or sum(int(value) for value in counts.values()) != judgeable
    ):
        raise ValueError("molecules judgment count totals differ")
    labels = judgment.get("stable_labels")
    if not isinstance(labels, dict) or len(labels) != judgeable:
        raise ValueError("molecules stable-label coverage differs")
    observed_counts = {
        label: sum(value == label for value in labels.values())
        for label in expected_labels
    }
    if observed_counts != {key: int(value) for key, value in counts.items()}:
        raise ValueError("molecules stable labels and counts differ")
    if not _close(
        judgment["q_positive_over_all_attempts"],
        int(counts["positive"]) / attempts,
    ):
        raise ValueError("molecules positive fraction has the wrong denominator")
    repeat_paths = judgment.get("repeat_artifacts")
    if not isinstance(repeat_paths, list) or len(repeat_paths) != 2:
        raise ValueError("molecules judgment does not contain two passes")
    if any(not Path(str(path)).is_file() for path in repeat_paths):
        raise ValueError("molecules judgment repeat artifact is missing")
    return judgment


def _validate_training(
    training: Mapping[str, Any], spec: Mapping[str, Any], *, stage: str
) -> None:
    if training.get("ltfd_stage") != stage:
        raise ValueError(f"{stage} update used the wrong DPO stage")
    if not (
        training.get("training_completed") is True
        and training.get("checkpoint_reload_verified") is True
        and training.get("originally_frozen_state_preserved") is True
        and training.get("all_pairs_visited") is True
        and training.get("parent_checkpoint_sha256")
        != training.get("child_checkpoint_sha256")
    ):
        raise ValueError(f"{stage} FlowMol training checks failed")

    hyperparameters = training.get("training_hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise ValueError(f"{stage} DPO hyperparameters are missing")
    expected_hyperparameters = {
        "learning_rate": float(spec["learning_rate"]),
        "optimizer_steps": int(spec["steps"]),
        "pair_microbatch_size": int(spec["batch_size_pairs"]),
        "gradient_accumulation_steps": int(spec["gradient_accumulation_steps"]),
        "effective_pair_batch_size": int(spec["effective_batch_size_pairs"]),
        "beta": float(spec["beta"]),
        "gradient_clip_norm": float(spec["gradient_clip_norm"]),
        "reference_policy": str(spec["reference_policy"]),
        "pair_construction": str(spec["pair_construction"]),
    }
    observed_without_pair_policy = {
        key: value
        for key, value in hyperparameters.items()
        if key != "pair_construction"
    }
    expected_without_pair_policy = {
        key: value
        for key, value in expected_hyperparameters.items()
        if key != "pair_construction"
    }
    if observed_without_pair_policy != expected_without_pair_policy:
        raise ValueError(f"{stage} DPO hyperparameters differ from the frozen profile")
    configured_construction = str(spec["pair_construction"])
    observed_construction = str(hyperparameters.get("pair_construction"))
    legacy_capped_metadata = (
        configured_construction.startswith(_CAPPED_CARTESIAN)
        and observed_construction == _FULL_CARTESIAN
    )
    if (
        observed_construction != configured_construction
        and not legacy_capped_metadata
    ):
        raise ValueError(f"{stage} DPO pair policy differs from the frozen profile")
    if int(training["optimizer_steps"]) != int(spec["steps"]):
        raise ValueError(f"{stage} optimizer-step count differs")
    if int(training["gradient_accumulation_steps"]) != int(
        spec["gradient_accumulation_steps"]
    ):
        raise ValueError(f"{stage} gradient accumulation differs")
    if int(training["microbatch_count"]) != int(spec["steps"]) * int(
        spec["gradient_accumulation_steps"]
    ):
        raise ValueError(f"{stage} microbatch count differs")

    pair_count = int(training["pair_count"])
    manifest = training.get("pair_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{stage} pair manifest is missing")
    positive_count = int(manifest["positive_count"])
    negative_count = int(manifest["negative_count"])
    cartesian_pair_count = positive_count * negative_count
    if configured_construction == _FULL_CARTESIAN:
        expected_manifest_construction = _FULL_CARTESIAN
        expected_pair_count = cartesian_pair_count
    elif configured_construction.startswith(_CAPPED_CARTESIAN):
        expected_manifest_construction = (_CAPPED_CARTESIAN if cartesian_pair_count > int(spec["maximum_pair_count"]) else _FULL_CARTESIAN)
        expected_pair_count = min(
            cartesian_pair_count, int(spec["maximum_pair_count"])
        )
    else:
        raise ValueError(f"{stage} uses an unsupported DPO pair policy")
    if (
        manifest.get("construction") != expected_manifest_construction
        or pair_count != expected_pair_count
        or pair_count != int(manifest["pair_count"])
        or int(manifest["cartesian_pair_count"]) != cartesian_pair_count
        or pair_count > int(spec["maximum_pair_count"])
        or int(training["visited_pair_count"]) != pair_count
        or int(training["minimum_pair_visit_count"]) < 1
    ):
        raise ValueError(f"{stage} did not train the registered Cartesian pair set")
    if not 0 < float(training["relative_parameter_delta"]) <= float(
        spec["maximum_relative_parameter_delta"]
    ):
        raise ValueError(f"{stage} FlowMol parameter change exceeded its bound")


def _noncollapse_valid(noncollapse: Mapping[str, Any]) -> bool:
    return noncollapse_valid(noncollapse)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_campaign_config(args.config.resolve())
    root = args.output_root.resolve()
    original_config, _snapshot = bind_configuration(config, root, create=False)
    result = read_json(root / "campaign_result.json")
    if result.get("experiment_id") != config["experiment_id"]:
        raise ValueError("molecules campaign result experiment differs")
    if result.get("gnina_used") is not False:
        raise ValueError("held-out GNINA leaked into adaptation")
    rounds = result.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("molecules campaign completed no curriculum round")
    if len(rounds) > min(
        int(config["curriculum_loop"]["maximum_alignment_rounds"]),
        int(config["geometric_control"]["maximum_total_updates"]),
    ):
        raise ValueError("molecules campaign exceeded its curriculum ceiling")

    p0_final = _validate_judgment(
        root, "F-p0", expected_goal=str(config["final_goal"])
    )
    if not _close(
        result["p0_final_goal_q"], p0_final["q_positive_over_all_attempts"]
    ):
        raise ValueError("campaign p0 F value differs from its judgment")

    discovery = config["curriculum_discovery"]
    control = config["geometric_control"]
    heuristic = config["heuristic_control"]
    validated_rounds = []
    previous_checkpoint_hash = None
    previous_goal_id = None
    consecutive_goal_updates = 0
    semantic_tightenings = 0
    final_reachability_streak = 0
    terminal_final = p0_final

    for expected_round, row in enumerate(rounds, 1):
        control = control_for_round(config, original_config, expected_round)
        if row.get("round") != expected_round:
            raise ValueError("molecules round sequence is not contiguous")
        training = read_json(Path(row["training_result"]))
        noncollapse = read_json(Path(row["noncollapse"]))
        if previous_checkpoint_hash is not None and training['parent_checkpoint_sha256'] != previous_checkpoint_hash:
            raise ValueError('checkpoint lineage is not contiguous')
        previous_checkpoint_hash = training['child_checkpoint_sha256']
        if noncollapse.get('parent_checkpoint_sha256') != training['parent_checkpoint_sha256'] or noncollapse.get('child_checkpoint_sha256') != training['child_checkpoint_sha256']:
            raise ValueError('non-collapse check is bound to another checkpoint pair')
        _validate_training(training, config["fine_tuning"], stage="curriculum")
        if not _noncollapse_valid(noncollapse):
            raise ValueError(f"molecules round {expected_round} non-collapse failed")

        entry = _validate_judgment(
            root, str(row["entry_judgment"]), expected_goal=str(row["goal"])
        )
        child = _validate_judgment(
            root,
            str(row["child_development_judgment"]),
            expected_goal=str(row["goal"]),
        )
        terminal_final = _validate_judgment(
            root,
            str(row["final_goal_monitor_judgment"]),
            expected_goal=str(config["final_goal"]),
        )
        if (
            not judgments_learnable(entry, discovery)
            or int(training["pair_manifest"]["positive_count"])
            != int(entry["counts"]["positive"])
            or int(training["pair_manifest"]["negative_count"])
            != int(entry["counts"]["negative"])
        ):
            raise ValueError("curriculum DPO entry did not match the stable-label gate")

        movement = row["movement"]
        if not _close(
            movement["parent_working_q"], entry["q_positive_over_all_attempts"]
        ) or not _close(
            movement["child_q"], child["q_positive_over_all_attempts"]
        ):
            raise ValueError("curriculum movement is not bound to its judgments")
        child_minus_adoption = float(movement["child_q"]) - float(
            movement["adoption_q"]
        )
        child_minus_parent = float(movement["child_q"]) - float(
            movement["parent_working_q"]
        )
        if not _close(
            movement["child_minus_adoption"], child_minus_adoption
        ) or not _close(
            movement["child_minus_parent_working"], child_minus_parent
        ):
            raise ValueError("curriculum movement delta is inconsistent")
        expected_enriched = goal_passed(control, movement)
        if bool(row["enriched"]) != expected_enriched:
            raise ValueError("curriculum enrichment decision is inconsistent")

        goal_action = row["goal_action"]
        if goal_action not in {"broaden", "tighten", "hold"}:
            raise ValueError("curriculum goal action differs")
        if goal_action != "hold":
            minimum_unlock = float(
                control[
                    "minimum_initial_broadening_gain"
                    if goal_action == "broaden"
                    else "minimum_new_goal_unlock"
                ]
            )
            if not clears_unlock(float(row["new_goal_unlock"]), minimum_unlock, config, expected_round):
                raise ValueError("accepted curriculum goal did not clear its unlock gate")
        if row["goal_id"] == previous_goal_id:
            consecutive_goal_updates += 1
            if goal_action != "hold":
                raise ValueError("repeated curriculum goal was not recorded as HOLD")
        else:
            consecutive_goal_updates = 1
            if previous_goal_id is None and goal_action != "broaden":
                raise ValueError("first curriculum goal was not a broadening")
            if previous_goal_id is not None:
                if goal_action != "tighten":
                    raise ValueError("new later curriculum goal was not a tightening")
                semantic_tightenings += 1
        if consecutive_goal_updates > int(control["maximum_updates_per_goal"]):
            raise ValueError("curriculum exceeded updates per goal")
        previous_goal_id = row["goal_id"]

        if not _close(
            row["final_goal_q"], terminal_final["q_positive_over_all_attempts"]
        ):
            raise ValueError("working-F monitor is not bound to its judgment")
        f_qualifies = (
            float(row["final_goal_q"]) >= float(heuristic["working_F_floor"])
            and float(row["final_goal_q"])
            - float(result["p0_final_goal_q"])
            >= float(heuristic["working_F_gain_over_p0"])
        )
        final_reachability_streak = final_reachability_streak + 1 if f_qualifies else 0
        if int(row["final_goal_reachability_streak"]) != final_reachability_streak:
            raise ValueError("working-F reachability streak is inconsistent")
        validated_rounds.append(
            {
                "round": expected_round,
                "training_completed": True,
                "noncollapse_passed": True,
                "enriched": bool(row["enriched"]),
            }
        )

    if semantic_tightenings > int(control["maximum_semantic_tightenings"]):
        raise ValueError("molecules campaign exceeded semantic-tightening ceiling")
    if not _close(
        result["curriculum_terminal_final_goal_q"],
        terminal_final["q_positive_over_all_attempts"],
    ):
        raise ValueError("campaign terminal curriculum F value differs")
    if result["curriculum_stop_reason"] == "working_final_goal_reachable" and (
        final_reachability_streak
        < int(heuristic["working_F_consecutive_cohorts"])
    ):
        raise ValueError("working-F stop rule was not satisfied")

    terminal_final = _validate_judgment(root, str(result["final_f_entry_judgment"]), expected_goal=str(config["final_goal"]))
    final_control = config["final_f_continuation"]
    expected_final_entry = (
        (
            not bool(final_control.get("requires_at_least_one_curriculum_update"))
            or bool(rounds)
        )
        and judgments_learnable(terminal_final, discovery, count_limits=final_control)
    )
    if bool(result["final_f_entry_eligible"]) != expected_final_entry:
        raise ValueError("final-F entry decision differs from the registered gate")
    final_rounds = result.get("final_f_updates", [])
    if not isinstance(final_rounds, list) or len(final_rounds) > int(
        final_control["maximum_updates"]
    ):
        raise ValueError("molecules campaign exceeded its final-F update ceiling")
    if final_rounds and not expected_final_entry:
        raise ValueError("final-F continuation bypassed its entry gate")
    if bool(final_control["enabled"]) and expected_final_entry and not final_rounds:
        raise ValueError("eligible final-F continuation did not run")

    validated_final_f = []
    previous_final_counts = dict(terminal_final["counts"])
    previous_final_q = float(terminal_final["q_positive_over_all_attempts"])
    for expected_round, row in enumerate(final_rounds, 1):
        if row.get("round") != expected_round or row.get("trained_goal_id") != "F":
            raise ValueError("final-F round sequence is invalid")
        training = read_json(Path(row["training_result"]))
        noncollapse = read_json(Path(row["noncollapse"]))
        if previous_checkpoint_hash is not None and training['parent_checkpoint_sha256'] != previous_checkpoint_hash:
            raise ValueError('checkpoint lineage is not contiguous')
        previous_checkpoint_hash = training['child_checkpoint_sha256']
        if noncollapse.get('parent_checkpoint_sha256') != training['parent_checkpoint_sha256'] or noncollapse.get('child_checkpoint_sha256') != training['child_checkpoint_sha256']:
            raise ValueError('non-collapse check is bound to another checkpoint pair')
        _validate_training(
            training, final_control["fine_tuning"], stage="final_f"
        )
        if not _noncollapse_valid(noncollapse):
            raise ValueError("final-F non-collapse validation failed")
        child = _validate_judgment(
            root,
            str(row["child_final_goal_judgment"]),
            expected_goal=str(config["final_goal"]),
        )
        entry = _validate_judgment(root, str(row["entry_final_goal_judgment"]), expected_goal=str(config["final_goal"]))
        if not judgments_learnable(entry, discovery, count_limits=final_control):
            raise ValueError("final-F entry did not clear the repeated-judgment gate")
        previous_final_counts = dict(entry["counts"])
        previous_final_q = float(entry["q_positive_over_all_attempts"])
        if (
            row["entry_counts"] != previous_final_counts
            or not _close(row["entry_q"], previous_final_q)
        ):
            raise ValueError("final-F updates are not a contiguous working chain")
        if (
            int(row["entry_counts"]["positive"])
            < int(final_control["minimum_stable_positive_count"])
            or int(row["entry_counts"]["negative"])
            < int(final_control["minimum_stable_negative_count"])
        ):
            raise ValueError("final-F entry did not clear its registered label gate")
        if (
            int(training["pair_manifest"]["positive_count"])
            != int(row["entry_counts"]["positive"])
            or int(training["pair_manifest"]["negative_count"])
            != int(row["entry_counts"]["negative"])
        ):
            raise ValueError("final-F DPO pair counts differ from entry labels")
        if row["child_counts"] != child["counts"] or not _close(
            row["child_q"], child["q_positive_over_all_attempts"]
        ):
            raise ValueError("final-F child record differs from its judgment")
        previous_final_counts = dict(row["child_counts"])
        previous_final_q = float(row["child_q"])
        validated_final_f.append(
            {"round": expected_round, "training_completed": True}
        )

    if len(rounds) + len(final_rounds) > int(
        config["registered_compute_budget"]["maximum_generator_updates"]
    ):
        raise ValueError("molecules campaign exceeded generator-update budget")
    if int(result["final_f_update_count"]) != len(final_rounds):
        raise ValueError("campaign final-F update count differs")
    terminal_working = _validate_judgment(root, str(result["terminal_working_final_goal_judgment"]), expected_goal=str(config["final_goal"]))
    if not _close(result["terminal_working_final_goal_q"], terminal_working["q_positive_over_all_attempts"]):
        raise ValueError("campaign terminal working-F value differs")

    calls = result["provider_calls"]
    for role in (
        "candidate_judge",
        "curriculum_controller",
        "independent_auditor",
    ):
        if int(calls[role]) > int(config[role]["maximum_provider_calls"]):
            raise ValueError(f"{role} exceeded its provider-call ceiling")
    receipt = {
        "schema_version": 1,
        "artifact_type": "molecules_ltfd_terminal_validation",
        "experiment_id": config["experiment_id"],
        "validated_rounds": validated_rounds,
        "completed_round_count": len(validated_rounds),
        "validated_final_f_updates": validated_final_f,
        "curriculum_stop_reason": result["curriculum_stop_reason"],
        "provider_calls": calls,
        "gnina_used_during_adaptation": result.get("gnina_used"),
        "artifact_chain_valid": True,
        "scientific_success_claimed": False,
    }
    immutable_json(root / "terminal_validation.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
