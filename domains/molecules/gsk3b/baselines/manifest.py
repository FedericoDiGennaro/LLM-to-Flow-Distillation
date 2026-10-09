#!/usr/bin/env python3
"""Bind a FlowMol rollout to its GSK3B rewards for a DiffusionNFT iteration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from domains.molecules.gsk3b.artifacts import immutable_json, read_json, sha256_file


def build_rollout(
    *,
    samples_dir: Path,
    arm: str,
    seed: int,
    collection_checkpoint: str,
    group_size: int | None,
) -> dict[str, Any]:
    if group_size is not None and group_size < 1:
        raise ValueError("group size must be at least one")
    import torch

    from domains.molecules.gsk3b.validate_samples import native_signature

    samples_path = samples_dir / "samples.pt"
    if not samples_path.is_file():
        raise FileNotFoundError(f"rollout has no samples.pt: {samples_path}")
    # This is a hash-bound FlowMol artifact produced by this campaign.  It
    # contains SampledMolecule instances rather than tensor-only weights, so
    # PyTorch >=2.6 must be told explicitly to use the trusted object loader.
    values = torch.load(samples_path, map_location="cpu", weights_only=False)
    molecules = values.get("molecules", [])
    if not isinstance(molecules, list) or not molecules:
        raise ValueError("FlowMol rollout has no molecules")
    digest = sha256_file(samples_path)
    sdf_dir = samples_dir / "sdf"
    samples: list[dict[str, Any]] = []
    for index, molecule in enumerate(molecules):
        sdf = sdf_dir / f"sample_{index}.sdf"
        assembled = sdf.is_file()
        group = index // group_size if group_size else 0
        samples.append(
            {
                "candidate_id": f"{arm}-s{seed}-{index:04d}",
                "group_id": f"{arm}-s{seed}-g{group:04d}",
                "sdf": str(sdf.resolve()) if assembled else None,
                "sdf_sha256": sha256_file(sdf) if assembled else None,
                "assembled": assembled,
                "content_sha256": native_signature(molecule),
                "training_state": {
                    "samples_pt": str(samples_path.resolve()),
                    "samples_pt_sha256": digest,
                    "sample_index": index,
                },
            }
        )
    return {
        "schema_version": 1,
        "artifact_type": "flowmol_nft_rollout",
        "arm": arm,
        "seed": seed,
        "collection_checkpoint": collection_checkpoint,
        "samples_pt": str(samples_path.resolve()),
        "samples_pt_sha256": digest,
        "sample_count": len(samples),
        "assembled_count": sum(1 for sample in samples if sample["assembled"]),
        "group_size": group_size or len(samples),
        "group_count": len({sample["group_id"] for sample in samples}),
        "samples": samples,
    }


def join_rewards(
    rollout: Mapping[str, Any], reward: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rewards = {str(record["candidate_id"]): record for record in reward["samples"]}
    bindings = {str(sample["candidate_id"]): sample for sample in rollout["samples"]}
    if len(rewards) != len(reward["samples"]) or len(bindings) != len(rollout["samples"]):
        raise ValueError("duplicate candidate identifiers in rollout/rewards")
    if not bindings:
        raise ValueError("empty reward/rollout cohort")
    if set(rewards) != set(bindings):
        missing = sorted(set(bindings) - set(rewards))
        extra = sorted(set(rewards) - set(bindings))
        raise ValueError(
            f"reward and rollout candidate sets differ (missing={missing}, extra={extra})"
        )
    joined: list[dict[str, Any]] = []
    for candidate_id, binding in bindings.items():
        record = rewards[candidate_id]
        if str(record["group_id"]) != str(binding["group_id"]):
            raise ValueError(f"{candidate_id} was rewarded under a different group")
        value = record["raw_reward"]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{candidate_id} has an invalid GSK3B probability")
        if type(record.get("scored_by_model")) is not bool:
            raise ValueError(f"{candidate_id} lacks a boolean scoring status")
        joined.append(
            {
                "candidate_id": candidate_id,
                "group_id": binding["group_id"],
                "raw_reward": float(record["raw_reward"]),
                "scored_by_model": bool(record["scored_by_model"]),
                "content_sha256": binding["content_sha256"],
                "training_state": binding["training_state"],
            }
        )
    return sorted(joined, key=lambda row: row["candidate_id"])


def build_manifest(
    *,
    rollout: Mapping[str, Any],
    reward: Mapping[str, Any],
    training_goal: str,
    immutable_final_goal: str,
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    # Keep this older API as a checked alias, not a second reward contract.
    from domains.molecules.gsk3b.baselines.nft_manifest import manifest
    from domains.molecules.gsk3b.oracle import REWARD_ENDPOINT

    if training_goal != REWARD_ENDPOINT:
        raise ValueError("GSK3B baseline training goal differs from its reward")
    if list(samples) != join_rewards(rollout, reward):
        raise ValueError("supplied training samples differ from the bound rollout")
    return manifest(rollout, reward, immutable_final_goal)


def nft_contract_from_campaign(
    config: Mapping[str, Any], *, collection_checkpoint: str,
) -> dict[str, Any]:
    from domains.molecules.gsk3b.baselines.nft_manifest import contract

    return contract(config, collection_checkpoint)



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)

    rollout_parser = sub.add_parser("rollout")
    rollout_parser.add_argument("--samples-dir", type=Path, required=True)
    rollout_parser.add_argument("--arm", default="nft")
    rollout_parser.add_argument("--seed", type=int, required=True)
    rollout_parser.add_argument("--collection-checkpoint", required=True)
    rollout_parser.add_argument("--group-size", type=int)
    rollout_parser.add_argument("--output", type=Path, required=True)

    manifest_parser = sub.add_parser("manifest")
    manifest_parser.add_argument("--rollout", type=Path, required=True)
    manifest_parser.add_argument("--reward", type=Path, required=True)
    manifest_parser.add_argument("--config", type=Path, required=True)
    manifest_parser.add_argument("--training-goal", required=True)
    manifest_parser.add_argument("--immutable-final-goal", required=True)
    manifest_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.phase == "rollout":
        if args.group_size is not None and args.group_size < 1:
            raise ValueError("group size must be at least one")
        rollout = build_rollout(
            samples_dir=args.samples_dir.resolve(),
            arm=args.arm,
            seed=args.seed,
            collection_checkpoint=args.collection_checkpoint,
            group_size=args.group_size,
        )
        immutable_json(args.output, rollout)
        print(
            json.dumps(
                {
                    "rollout": str(args.output.resolve()),
                    "sample_count": rollout["sample_count"],
                    "assembled_count": rollout["assembled_count"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    from domains.molecules.gsk3b.baselines.nft_loss import validate_nft_preflight
    from domains.molecules.gsk3b.config import load_campaign_config

    rollout = read_json(args.rollout)
    reward = read_json(args.reward)
    config = load_campaign_config(args.config.resolve())
    contract = nft_contract_from_campaign(
        config, collection_checkpoint=str(rollout["collection_checkpoint"])
    )
    samples = join_rewards(rollout, reward)
    manifest = build_manifest(
        rollout=rollout,
        reward=reward,
        training_goal=args.training_goal,
        immutable_final_goal=args.immutable_final_goal,
        samples=samples,
    )
    check = validate_nft_preflight(contract, manifest)
    if not check.ready:
        raise ValueError(str(check.blockers))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    immutable_json(output_dir / "contract.json", contract)
    immutable_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "contract": str(output_dir / "contract.json"),
                "manifest": str(output_dir / "manifest.json"),
                "sample_count": len(samples),
                "scored_count": manifest["scored_count"],
                "floored_count": manifest["floored_count"],
                "reward_mean": manifest["reward_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
