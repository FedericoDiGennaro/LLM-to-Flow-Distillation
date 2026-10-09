#!/usr/bin/env python3
"""Generate preregistered direct-reward baselines from released PepDFM p0."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
ARMS = (
    "unguided_p0",
    "pool_tilt_acp",
    "pool_tilt_acp_safe",
    "local_guidance_acp",
    "local_guidance_acp_safe",
)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path | str):
    return json.loads(Path(path).read_text())


def load_config(path: Path | str) -> dict:
    """Load a portable config, resolving source-controlled inputs from it."""
    path = Path(path).resolve()
    config = read_json(path)
    for key in ("source_root", "acp_parameters", "hemolysis_parameters",
                "hemopi2_adapter", "endpoint_control_candidates"):
        value = config.get(key)
        if value and not Path(value).is_absolute():
            config[key] = str((path.parent / value).resolve())
    config["asset_dir"] = os.environ.get("PEPDFM_ASSET_DIR", config["asset_dir"])
    config["released_hemopi2_assets"] = os.environ.get(
        "HEMOPI2_ASSETS", config["released_hemopi2_assets"])
    config["released_hemopi2_overlay"] = os.environ.get(
        "HEMOPI2_OVERLAY", config["released_hemopi2_overlay"])
    config["anticp2_assets"] = os.environ.get(
        "ANTICP2_ASSETS", config["anticp2_assets"])
    for key in ("asset_dir", "released_hemopi2_assets", "released_hemopi2_overlay",
                "anticp2_assets"):
        value = Path(config[key])
        if not value.is_absolute():
            config[key] = str((path.parent / value).resolve())
    return config


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def check(path: Path | str, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")


def feature_vector(sequence: str) -> np.ndarray:
    """The exact frozen AAC20+DPC400 feature definition used in EXP-0106."""
    if not sequence or not set(sequence) <= set(AA):
        raise ValueError("noncanonical peptide")
    aa_index = {aa: i for i, aa in enumerate(AA)}
    values = np.zeros(420, dtype=np.float64)
    for residue in sequence:
        values[aa_index[residue]] += 1.0 / len(sequence)
    if len(sequence) > 1:
        for left, right in zip(sequence, sequence[1:]):
            values[20 + aa_index[left] * 20 + aa_index[right]] += 1.0 / (len(sequence) - 1)
    return values


def probability(sequence: str, params: dict) -> float:
    features = feature_vector(sequence)
    z = float(((features - np.asarray(params["mean"])) / np.asarray(params["scale"]))
              @ np.asarray(params["coefficients"]) + params["intercept"])
    return 1.0 / (1.0 + math.exp(-z))


def seed(config: dict, replicate: int, stream: int, length: int) -> int:
    return int(config["replicate_seeds"][replicate]) + stream * 100_000 + length * 1_000


def sample_rows(result, arm: str, replicate: int, length: int, stream_seed: int):
    return [{**asdict(sample), "arm_id": arm, "replicate_index": replicate,
             "requested_length": length, "sampling_seed": stream_seed,
             "draw_id": f"r{replicate}-{arm}-L{length}-{sample.sample_index:04d}"}
            for sample in result.samples]


def resample_pool(rows: list[dict], params: dict, safe_params: dict | None,
                  count: int, resampling_seed: int, arm: str) -> tuple[list[dict], dict]:
    scores = []
    for row in rows:
        if not row["is_valid"]:
            scores.append(0.0)
            continue
        acp = probability(row["sequence"], params)
        scores.append(acp if safe_params is None else acp * (1.0 - probability(row["sequence"], safe_params)))
    weights = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid or empty pool weights")
    normalized = weights / weights.sum()
    rng = np.random.default_rng(resampling_seed)
    selected = rng.choice(len(rows), size=count, replace=True, p=normalized)
    output = []
    for index, source_index in enumerate(selected):
        source = rows[int(source_index)]
        output.append({**source, "arm_id": arm,
                       "draw_id": f"r{source['replicate_index']}-{arm}-L{source['requested_length']}-{index:04d}",
                       "source_pool_draw_id": source["draw_id"],
                       "source_pool_index": int(source_index),
                       "source_pool_probability": float(normalized[source_index]),
                       "resampling_seed": resampling_seed})
    positive = normalized[normalized > 0]
    diagnostic = {
        "arm_id": arm, "requested_count": count, "pool_count": len(rows),
        "valid_pool_count": sum(row["is_valid"] for row in rows),
        "resampling_seed": resampling_seed,
        "normalized_weight_ess": float(1.0 / np.square(normalized).sum()),
        "normalized_weight_ess_fraction": float(1.0 / np.square(normalized).sum() / len(rows)),
        "maximum_normalized_weight": float(normalized.max()),
        "minimum_positive_normalized_weight": float(positive.min()),
        "selected_unique_source_draws": len(set(int(i) for i in selected)),
        "duplicate_multiplicity_preserved": True,
        "source_weights": [
            {"source_pool_draw_id": row["draw_id"],
             "unnormalized_reward_weight": float(weight),
             "normalized_weight": float(normalized_weight)}
            for row, weight, normalized_weight in zip(rows, weights, normalized)
        ],
        "selected_source_pool_indices": [int(index) for index in selected],
    }
    return output, diagnostic


def make_objective(torch, params: dict, safe_params: dict | None, name: str):
    token_by_id = ("<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E",
                   "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C")
    feature_index = torch.full((24,), -1, dtype=torch.long)
    for token_id, residue in enumerate(token_by_id):
        if residue in AA:
            feature_index[token_id] = AA.index(residue)

    class FrozenAacDpcReward(torch.nn.Module):
        objective_name = name

        def __init__(self):
            super().__init__()
            self.register_buffer("feature_index", feature_index)
            arrays = [params["mean"], params["scale"], params["coefficients"]]
            self.register_buffer("acp_mean", torch.tensor(arrays[0], dtype=torch.float32))
            self.register_buffer("acp_scale", torch.tensor(arrays[1], dtype=torch.float32))
            self.register_buffer("acp_coef", torch.tensor(arrays[2], dtype=torch.float32))
            self.register_buffer("acp_intercept", torch.tensor(params["intercept"], dtype=torch.float32))
            if safe_params is not None:
                self.register_buffer("safe_mean", torch.tensor(safe_params["mean"], dtype=torch.float32))
                self.register_buffer("safe_scale", torch.tensor(safe_params["scale"], dtype=torch.float32))
                self.register_buffer("safe_coef", torch.tensor(safe_params["coefficients"], dtype=torch.float32))
                self.register_buffer("safe_intercept", torch.tensor(safe_params["intercept"], dtype=torch.float32))

        def features(self, tokens):
            residue_tokens = tokens[:, 1:-1]
            indexes = self.feature_index[residue_tokens]
            valid = (indexes >= 0).all(dim=1)
            # Upstream scores all vocabulary substitutions before masking token
            # IDs 0--3. Compute harmless features for those temporary rows and
            # return a finite floor; canonical candidate scores are unchanged.
            safe_indexes = indexes.clamp_min(0)
            aa = torch.nn.functional.one_hot(safe_indexes, 20).float().mean(dim=1)
            pairs = safe_indexes[:, :-1] * 20 + safe_indexes[:, 1:]
            dpc = torch.nn.functional.one_hot(pairs, 400).float().mean(dim=1)
            return torch.cat((aa, dpc), dim=1), valid

        @staticmethod
        def log_probability(features, mean, scale, coef, intercept):
            z = ((features - mean) / scale) @ coef + intercept
            return torch.nn.functional.logsigmoid(z)

        def forward(self, tokens):
            features, valid = self.features(tokens)
            reward = self.log_probability(features, self.acp_mean, self.acp_scale,
                                          self.acp_coef, self.acp_intercept)
            if safe_params is not None:
                z = ((features - self.safe_mean) / self.safe_scale) @ self.safe_coef + self.safe_intercept
                reward = reward + torch.nn.functional.logsigmoid(-z)
            return torch.where(valid, reward, torch.full_like(reward, -1.0e6))

    return FrozenAacDpcReward()


def audit(config: dict) -> None:
    if tuple(config["arms"]) != ARMS or config["beta"] != 1.0 or config["pool_multiplier"] != 10:
        raise ValueError("frozen arm, beta, or pool protocol changed")
    if {int(k): int(v) for k, v in config["length_counts"].items()} != {12: 35, 24: 40, 38: 25}:
        raise ValueError("length policy changed")
    if len(config["replicate_seeds"]) != 3 or len(set(config["replicate_seeds"])) != 3:
        raise ValueError("exactly three distinct sampling replicates required")
    for path, expected in config["pinned_files"].items():
        check(REPO_ROOT / path, expected)
    for path, expected in config["asset_pinned_files"].items():
        check(Path(config["asset_dir"]) / path, expected)
    for path, expected in config["anticp2_pinned_files"].items():
        check(Path(config["anticp2_assets"]) / path, expected)


def generate(config: dict, config_path: Path, output: Path, replicate: int) -> None:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("generation requires a tracked GPU Slurm step")
    audit(config)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "started.json", {
        "experiment_id": config["experiment_id"], "replicate_index": replicate,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "config_sha256": sha256(config_path),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "credentials_recorded": False,
    })
    sys.path.insert(0, config["source_root"])
    import torch
    from domains.peptides.model import PepDfmAssets, PepDfmSamplingConfig, sample_pepdfm
    from domains.peptides.guidance import MogGuidanceConfig, sample_pepdfm_with_mog_guidance
    if torch.__version__ != config["expected_torch_version"] or not torch.cuda.is_available():
        raise RuntimeError(f"unexpected runtime torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    write_new(output / "environment.json", {
        "python": sys.version, "python_executable": sys.executable,
        "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "cudnn": torch.backends.cudnn.version(),
        "source_root": config["source_root"], "asset_dir": config["asset_dir"],
        "pinned_files": config["pinned_files"],
        "environment_variables_recorded": False,
    })
    assets = PepDfmAssets.resolve(Path(config["asset_dir"]))
    acp, hemo = read_json(config["acp_parameters"]), read_json(config["hemolysis_parameters"])
    acp_objective = make_objective(torch, acp, None, "frozen_ridge_log_acp_probability")
    safe_objective = make_objective(torch, acp, hemo, "frozen_ridge_log_acp_plus_log_nonhemolysis")
    rows, diagnostics = [], []
    for length_string, count in sorted(config["length_counts"].items(), key=lambda pair: int(pair[0])):
        length, count = int(length_string), int(count)
        unguided_seed = seed(config, replicate, 0, length)
        call_started = time.monotonic()
        direct = sample_pepdfm(assets, PepDfmSamplingConfig(length=length, num_samples=count,
            num_steps=config["num_steps"], seed=unguided_seed, source_distribution="uniform", device="cuda:0"))
        direct_rows = sample_rows(direct, "unguided_p0", replicate, length, unguided_seed)
        rows.extend(direct_rows)
        write_new(output / f"raw_unguided_p0_L{length}.json", direct_rows)
        write_new(output / f"runtime_unguided_p0_L{length}.json", {
            "arm_id": "unguided_p0", "length": length, "draw_count": count,
            "generator_model_forward_calls": config["num_steps"], "reward_forward_calls": 0,
            "elapsed_seconds": time.monotonic() - call_started})
        pool_seed = seed(config, replicate, 1, length)
        call_started = time.monotonic()
        pool = sample_pepdfm(assets, PepDfmSamplingConfig(length=length,
            num_samples=count * config["pool_multiplier"], num_steps=config["num_steps"],
            seed=pool_seed, source_distribution="uniform", device="cuda:0"))
        pool_rows = sample_rows(pool, "proposal_pool", replicate, length, pool_seed)
        write_new(output / f"raw_proposal_pool_L{length}.json", pool_rows)
        pool_elapsed = time.monotonic() - call_started
        for arm, safe_params, stream in (("pool_tilt_acp", None, 2), ("pool_tilt_acp_safe", hemo, 3)):
            call_started = time.monotonic()
            selected, diagnostic = resample_pool(pool_rows, acp, safe_params, count,
                                                seed(config, replicate, stream, length), arm)
            diagnostic.update(shared_proposal_pool=True,
                shared_pool_generation_elapsed_seconds=pool_elapsed,
                generator_model_forward_calls_shared=config["num_steps"],
                reward_sequence_evaluations=len(pool_rows),
                resampling_elapsed_seconds=time.monotonic() - call_started)
            rows.extend(selected); diagnostics.append(diagnostic)
            write_new(output / f"raw_{arm}_L{length}.json", selected)
            write_new(output / f"pool_diagnostic_{arm}_L{length}.json", diagnostic)
        for arm, objective, stream in (("local_guidance_acp", acp_objective, 4),
                                       ("local_guidance_acp_safe", safe_objective, 5)):
            guided_seed = seed(config, replicate, stream, length)
            call_started = time.monotonic()
            result = sample_pepdfm_with_mog_guidance(assets,
                MogGuidanceConfig(length=length, num_samples=count, num_steps=config["num_steps"],
                    seed=guided_seed, source_distribution="uniform", device="cuda:0", num_div=64,
                    lambda_=1.0, beta=config["beta"], alpha_r=0.5, eta=1.0,
                    Phi_init=math.radians(45), Phi_min=math.radians(15),
                    Phi_max=math.radians(75), tau=0.3), [objective])
            guided_rows = sample_rows(result, arm, replicate, length, guided_seed)
            rows.extend(guided_rows)
            write_new(output / f"raw_{arm}_L{length}.json", guided_rows)
            write_new(output / f"runtime_{arm}_L{length}.json", {
                "arm_id": arm, "length": length, "draw_count": count,
                "generator_model_forward_calls": config["num_steps"],
                "reward_forward_calls": 2 * (config["num_steps"] - 1),
                "reward_sequence_evaluations": (config["num_steps"] - 1) * count * 24,
                "elapsed_seconds": time.monotonic() - call_started,
                "algorithm_note": "Released greedy single-position adaptive-hypercone heuristic; not exact sampling from a tilted p0 density."})
    expected = len(ARMS) * sum(config["length_counts"].values())
    if len(rows) != expected:
        raise ValueError(f"wrong cohort size {len(rows)} != {expected}")
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm_id"] == arm]
        if len(arm_rows) != 100 or any(not row["is_valid"] for row in arm_rows):
            raise ValueError(f"{arm} did not yield exactly 100 valid retained draws")
    write_new(output / "cohorts.json", rows)
    write_new(output / "pool_diagnostics.json", diagnostics)
    write_new(output / "completion.json", {"completed": True, "replicate_index": replicate,
        "draw_count": len(rows), "all_valid": True, "cohorts_sha256": sha256(output / "cohorts.json"),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()})


def merge(config: dict, config_path: Path, output: Path) -> None:
    audit(config)
    rows, pool = [], []
    for replicate in range(3):
        root = output / f"replicate-{replicate}"
        completion = read_json(root / "completion.json")
        if completion.get("completed") is not True:
            raise ValueError(f"replicate {replicate} incomplete")
        check(root / "cohorts.json", completion["cohorts_sha256"])
        rows += read_json(root / "cohorts.json")
        pool += read_json(root / "pool_diagnostics.json")
    if len(rows) != 1500 or len({row["draw_id"] for row in rows}) != 1500:
        raise ValueError("merged cohort must have 1500 unique draw IDs")
    write_new(output / "all_cohorts.json", rows)
    write_new(output / "all_pool_diagnostics.json", pool)
    write_new(output / "generation_completion.json", {"completed": True, "draw_count": len(rows),
        "all_cohorts_sha256": sha256(output / "all_cohorts.json"), "config_sha256": sha256(config_path),
        "sampling_replicates_are_not_training_replicates": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "generate", "merge"))
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicate-index", type=int, choices=range(3))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "audit":
        audit(config); print(json.dumps({"audit": "passed"}))
    elif args.command == "generate":
        if args.replicate_index is None:
            parser.error("generate requires --replicate-index")
        generate(config, args.config.resolve(), args.output, args.replicate_index)
    else:
        merge(config, args.config.resolve(), args.output)


if __name__ == "__main__":
    main()
