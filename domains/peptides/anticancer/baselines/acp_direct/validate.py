#!/usr/bin/env python3
"""Score direct-reward cohorts without changing any reward or sampling setting."""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from domains.peptides.anticancer.baselines.acp_direct import prepare as p
from domains.peptides.anticancer.baselines.evaluation import anticp2_frozen


def descriptors(sequence: str) -> dict[str, float]:
    length = len(sequence)
    counts = Counter(sequence)
    frequencies = np.asarray([counts[aa] / length for aa in p.AA], dtype=float)
    entropy = -float(sum(value * np.log(value) for value in frequencies if value > 0)) / np.log(20)
    return {
        "net_charge_approx": float(counts["K"] + counts["R"] + 0.1 * counts["H"] - counts["D"] - counts["E"]),
        "basic_fraction": float((counts["K"] + counts["R"] + counts["H"]) / length),
        "hydrophobic_fraction": float(sum(counts[aa] for aa in "AILMFWVY") / length),
        "aromatic_fraction": float(sum(counts[aa] for aa in "FWY") / length),
        "normalized_shannon_complexity": entropy,
    }


def diversity(rows: list[dict]) -> dict:
    sequences = [row["sequence"] for row in rows]
    by_length = {}
    weighted_hamming = 0.0
    for length, count in ((12, 35), (24, 40), (38, 25)):
        values = [sequence for sequence in sequences if len(sequence) == length]
        if len(values) != count:
            raise ValueError("fixed length composition changed")
        pairs = count * (count - 1) // 2
        mismatches = 0
        for position in zip(*values):
            mismatches += pairs - sum(n * (n - 1) // 2 for n in Counter(position).values())
        distance = mismatches / (pairs * length)
        by_length[str(length)] = {"draw_count": count, "unique_count": len(set(values)),
                                  "mean_pairwise_normalized_hamming": distance}
        weighted_hamming += count / 100 * distance
    return {"unique_count": len(set(sequences)), "unique_fraction": len(set(sequences)) / 100,
            "fixed_length_weighted_mean_pairwise_normalized_hamming": weighted_hamming,
            "by_length": by_length}


def describe(values) -> dict:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("nonfinite metric")
    return {"mean": float(array.mean()), "sd_across_draws": float(array.std(ddof=1)),
            "minimum": float(array.min()), "maximum": float(array.max())}


def run_hemopi(config: dict, rows: list[dict], output: Path) -> dict[str, dict]:
    fasta = output / "candidates.fasta"
    with fasta.open("x") as handle:
        for row in rows:
            handle.write(f">{row['draw_id']}\n{row['sequence']}\n")
    adapter_output = output / "released_hemopi2_adapter"
    command = [sys.executable, config["hemopi2_adapter"], "--input-fasta", str(fasta),
               "--assets", config["released_hemopi2_assets"], "--output", str(adapter_output),
               "--mode", "classification", "--threshold", "0.45", "--allow-model-load"]
    environment = dict(os.environ)
    environment.update(PYTHONPATH=config["released_hemopi2_overlay"], PYTHONNOUSERSITE="1",
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (output / "hemopi2.stdout.log").open("x") as stdout, (output / "hemopi2.stderr.log").open("x") as stderr:
        subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, check=True)
    completion = p.read_json(adapter_output / "completion.json")
    if completion.get("complete") is not True or completion.get("input_count") != len(rows):
        raise ValueError("released HemoPI2 did not complete every candidate")
    if completion.get("input_sha256") != p.sha256(fasta):
        raise ValueError("released HemoPI2 input hash differs")
    expected_adapter = config["pinned_files"]["domains/peptides/anticancer/baselines/evaluation/hemopi2_rf_adapter.py"]
    if completion.get("adapter_sha256") != expected_adapter:
        raise ValueError("released HemoPI2 adapter hash differs")
    p.check(adapter_output / "predictions.json", completion["predictions_sha256"])
    predictions = p.read_json(adapter_output / "predictions.json")
    indexed = {row["row_id"]: row for row in predictions}
    if (len(predictions) != len(rows) or len(indexed) != len(predictions)
            or set(indexed) != {row["draw_id"] for row in rows}):
        raise ValueError("released HemoPI2 candidate IDs differ")
    for source in rows:
        if indexed[source["draw_id"]].get("sequence") != source["sequence"]:
            raise ValueError("released HemoPI2 sequence changed under a retained ID")
    return indexed


def aggregate(replicates: dict[str, list[dict]], metrics: tuple[str, ...]) -> dict:
    output = {}
    baseline = replicates["unguided_p0"]
    for arm, records in replicates.items():
        if len(records) != 3:
            raise ValueError(f"{arm} requires three sampling replicates")
        item = {"sampling_replicates": records}
        for metric in metrics:
            values = np.asarray([record[metric] for record in records], dtype=float)
            item[metric] = {"mean_across_sampling_replicates": float(values.mean()),
                            "sd_across_sampling_replicates": float(values.std(ddof=1)),
                            "replicate_values": values.tolist()}
            if arm != "unguided_p0":
                base = np.asarray([record[metric] for record in baseline], dtype=float)
                delta = values - base
                item[metric].update(mean_paired_change_vs_unguided=float(delta.mean()),
                                    sd_paired_change_vs_unguided=float(delta.std(ddof=1)),
                                    paired_changes_vs_unguided=delta.tolist())
        output[arm] = item
    return output


def run(config_path: Path, generation: Path, output: Path) -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real scoring requires tracked Slurm")
    config = p.load_config(config_path); p.audit(config)
    completion = p.read_json(generation / "generation_completion.json")
    p.check(generation / "all_cohorts.json", completion["all_cohorts_sha256"])
    rows = p.read_json(generation / "all_cohorts.json")
    if len(rows) != 1500 or any(not row["is_valid"] for row in rows):
        raise ValueError("expected all 1500 preregistered valid draws")
    output.mkdir(parents=True, exist_ok=False)
    p.write_new(output / "started.json", {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"], "config_sha256": p.sha256(config_path),
        "cohorts_sha256": completion["all_cohorts_sha256"], "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    acp, hemo = p.read_json(config["acp_parameters"]), p.read_json(config["hemolysis_parameters"])
    released = run_hemopi(config, rows, output)
    alternate = anticp2_frozen.load_legacy_forest(
        Path(config["anticp2_assets"]) / "aac_extra_model", 20)
    alternate_scores = anticp2_frozen.legacy_predict_proba(
        alternate, anticp2_frozen.aac_features([row["sequence"] for row in rows]))[:, 1]
    enriched = []
    for row, alternate_probability in zip(rows, alternate_scores):
        sequence = row["sequence"]
        acp_probability, hemo_probability = p.probability(sequence, acp), p.probability(sequence, hemo)
        released_row = released[row["draw_id"]]
        if released_row.get("status") != "eligible":
            raise ValueError(f"released HemoPI2 rejected registered sequence: {row['draw_id']}")
        enriched.append({**row, **descriptors(sequence), "acp_proxy_probability": acp_probability,
            "hemolysis_proxy_probability": hemo_probability,
            "acp_safe_product": acp_probability * (1.0 - hemo_probability),
            "anticp2_alternate_probability": float(alternate_probability),
            "released_hemopi2_probability": float(released_row["hemolysis_probability_raw"])})
    p.write_new(output / "candidates.json", enriched)
    metric_names = ("acp_proxy_probability", "hemolysis_proxy_probability", "acp_safe_product",
                    "anticp2_alternate_probability",
                    "released_hemopi2_probability", "net_charge_approx", "basic_fraction",
                    "hydrophobic_fraction", "aromatic_fraction", "normalized_shannon_complexity",
                    "unique_fraction", "fixed_length_weighted_mean_pairwise_normalized_hamming")
    replicate_records: dict[str, list[dict]] = {arm: [] for arm in p.ARMS}
    cohort_summaries = []
    for arm in p.ARMS:
        for replicate in range(3):
            cohort = [row for row in enriched if row["arm_id"] == arm and row["replicate_index"] == replicate]
            if len(cohort) != 100:
                raise ValueError("cohort cardinality changed")
            diversity_record = diversity(cohort)
            record = {"arm_id": arm, "replicate_index": replicate,
                      **{metric: describe([row[metric] for row in cohort])["mean"]
                         for metric in metric_names[:10]},
                      "unique_fraction": diversity_record["unique_fraction"],
                      "fixed_length_weighted_mean_pairwise_normalized_hamming":
                          diversity_record["fixed_length_weighted_mean_pairwise_normalized_hamming"]}
            replicate_records[arm].append(record)
            cohort_summaries.append({**record, "diversity": diversity_record,
                "draw_metric_summaries": {metric: describe([row[metric] for row in cohort])
                                           for metric in metric_names[:10]}})
    report = {
        "schema_version": 1, "experiment_id": config["experiment_id"], "completed": True,
        "primary_metric": "exact-final-goal stable-positive fraction is evaluated in the subsequent frozen LLM stage",
        "mechanism_metric": "mean frozen ACP probability; direct reward target, therefore not independent validation",
        "safety_metrics": ["frozen ridge hemolysis probability (used in safe reward)",
                           "released HemoPI2 probability (not optimized, but sharing its provider data source)"],
        "claim_boundary": "Sequence-only computational diagnostics. ACP and hemolysis outputs are predictions, not biological efficacy or safety. The ridge ACP metric is the optimized reward and cannot validate itself. Sampling replicates are not training replicates.",
        "cohorts": cohort_summaries, "aggregate": aggregate(replicate_records, metric_names),
        "pool_diagnostics": p.read_json(generation / "all_pool_diagnostics.json"),
        "all_draws_and_duplicate_multiplicity_preserved": True,
    }
    p.write_new(output / "report.json", report)
    p.write_new(output / "completion.json", {"completed": True, "report_sha256": p.sha256(output / "report.json"),
        "candidates_sha256": p.sha256(output / "candidates.json"),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.generation.resolve(), args.output.resolve())
    print(json.dumps({"completed": result["completed"], "experiment_id": result["experiment_id"]}))


if __name__ == "__main__":
    main()
