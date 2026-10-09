"""CPP table baselines: DPO, DiffusionNFT, FlowGRPO and guidance.

Numerical kernels are preserved from the recorded baseline source.
"""

from __future__ import annotations


from pathlib import Path
from collections import Counter
import hashlib
import re
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
COMMIT = "4708e09d5a28cc9d843b3bf70c7d1ec93e79d881"


def features(sequences):
    out = []
    for s in sequences:
        if len(s) < 2 or not set(s) <= set(AA):
            raise ValueError("RF requires canonical full-length sequence")
        a, d = Counter(s), Counter(s[i : i + 2] for i in range(len(s) - 1))
        out.append(
            [a[c] / len(s) for c in AA]
            + [d[a + b] / (len(s) - 1) for a in AA for b in AA]
        )
    return np.asarray(out, dtype=np.float64)


def sequences(path):
    # Source files are RTF with one sequence per paragraph. Strip only RTF
    # controls, never residues, and assert the published counts downstream.
    text = Path(path).read_text()
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)
    return re.findall(r"(?<![A-Za-z])[" + AA + r"]{2,}(?![A-Za-z])", text)


def fit_reward(source):
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    import subprocess

    source = Path(source)
    if (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        != COMMIT
    ):
        raise ValueError("CellPPD source revision changed")
    names = [
        "Dataset1/positive_peptides.rtf",
        "Dataset1/negative_peptides.rtf",
        "Independent Set/positive.rtf",
        "Independent Set/negative.rtf",
    ]
    paths = [source / "CPP Datasets" / n for n in names]
    groups = [sequences(p) for p in paths]
    if [len(s) for s in groups] != [708, 708, 99, 99]:
        raise ValueError("CellPPD source counts differ")
    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=18601,
        n_jobs=4,
    )
    model.fit(features(groups[0] + groups[1]), np.r_[np.ones(708), np.zeros(708)])
    check = features(groups[2] + groups[3])
    pred = model.predict_proba(check)[:, 1]
    if not np.array_equal(model.classes_, [0, 1]):
        raise ValueError("class order differs")
    if not np.allclose(
        pred[:3],
        np.array([model.predict_proba(x[None])[:, 1][0] for x in check[:3]]),
        atol=1e-12,
    ):
        raise ValueError("RF batching parity failed")
    labels = np.r_[np.ones(99), np.zeros(99)]
    return model, {
        "source_commit": COMMIT,
        "sklearn_version": sklearn.__version__,
        "dataset_sha256": {
            n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in zip(names, paths)
        },
        "counts": [len(s) for s in groups],
        "diagnostic_auroc": float(roc_auc_score(labels, pred)),
        "diagnostic_balanced_accuracy": float(
            balanced_accuracy_score(labels, pred > 0.5)
        ),
        "train_diagnostic_exact_overlap": len(
            set(groups[0] + groups[1]) & set(groups[2] + groups[3])
        ),
        "validation_predictions": pred.tolist(),
        "hyperparameter_search": False,
        "caveat": "New RF, not released CellPPD SVM. Random negatives. All1416 training records overlap pLM4CPPs historical data.",
    }


class Reward:
    def __init__(self, model):
        self.model, self.calls = model, 0

    def __call__(self, seqs):
        self.calls += len(seqs)
        return self.model.predict_proba(features(seqs))[:, 1]


import torch
import numpy as np


def transition(q, x, t, h=0.01, final=False):
    t = torch.as_tensor(t, device=q.device, dtype=q.dtype)
    a = torch.ones_like(t) if final else -torch.expm1(-h * 2 * t / (1 - t * t))
    while a.ndim < q.ndim:
        a = a.unsqueeze(-1)
    delta = torch.nn.functional.one_hot(x, q.shape[-1]).to(q.dtype)
    return a * q + (1 - a) * delta


def log_probability(p, y):
    return p.gather(-1, y.unsqueeze(-1)).squeeze(-1).clamp_min(1e-30).log().sum(-1)


def guided_site(p, rewards, strength):
    # Stable positive reweighting preserves zero support and strength0 identity.
    w = torch.exp(strength * (rewards - rewards.max(-1, keepdim=True).values))
    tilted = p * w
    return tilted / tilted.sum(-1, keepdim=True)


def nft_optimality(rewards, lengths):
    rewards, lengths = np.asarray(rewards), np.asarray(lengths)
    out = np.full(len(rewards), 0.5)
    sd = rewards.std()
    if sd > 1e-8:
        for length in np.unique(lengths):
            idx = lengths == length
            out[idx] += 0.5 * np.clip((rewards[idx] - rewards[idx].mean()) / sd, -1, 1)
    return out


def nft_loss(student, old, target, reward, beta=1.0):
    old = old.detach()

    def branch(pred):
        err = pred - target
        dims = tuple(range(1, err.ndim))
        return err.square().mean(dims) / err.abs().mean(dims).detach().clamp_min(1e-5)

    return reward * branch((1 - beta) * old + beta * student) + (1 - reward) * branch(
        (1 + beta) * old - beta * student
    )


def ppo_loss(logp, oldlogp, advantage, clip=0.2):
    ratio = (logp - oldlogp.detach()).exp()
    advantage = advantage.detach()
    return -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    ).mean()


import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch

LENGTHS = {12: 35, 24: 40, 38: 25}
SEED = 18601
TOKENS = "LAGVSERTIDPKQNFYM HWC".replace(" ", "")
NFT_ARMS = ("nft7",)


def update_count(arm):
    return {"dpo": 1, "nft7": 7, "flowgrpo5": 5, "guidance": 0}[arm]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, path)
    temp.unlink()


def read(path):
    return json.loads(Path(path).read_text())


def validate_rows(rows):
    if len(rows) != 100:
        raise ValueError("expected100 draws")
    for r in rows:
        if (
            not r["is_valid"]
            or not r["sequence"]
            or len(r["sequence"]) != r["requested_length"]
            or not set(r["sequence"]) <= set(AA)
        ):
            raise ValueError(
                "invalid draw saved; no replacement; endpoint evaluation incomplete"
            )


def vanilla_draw(model, seed, label):
    rows = []
    for length, count in LENGTHS.items():
        for s in model.sample(
            num_samples=count,
            length=length,
            num_steps=100,
            seed=seed + length,
            batch_size=count,
        ):
            rows.append(
                {
                    **asdict(s),
                    "requested_length": length,
                    "draw_id": f"{label}-L{length}-{s.sample_index}",
                }
            )
    return rows


def discrete_draw(model, seed, label, reward=None, strength=0.0, keep=False):
    """Exact marginalized upstream transition. Different RNG from two-stage solver."""
    from domains.peptides.model import _decode_sample

    model.model.eval()
    rows = []
    traces = []
    with torch.random.fork_rng(devices=[model.device]), torch.no_grad():
        for length, count in LENGTHS.items():
            torch.manual_seed(seed + length)
            x = torch.randint(4, 24, (count, length + 2), device=model.device)
            x[:, 0] = 0
            x[:, -1] = 2
            xs = []
            ys = []
            logs = []
            for step in range(100):
                t = torch.full((count,), step * 0.01, device=model.device)
                q = model.model(x, t).softmax(-1)
                p = transition(q, x, t, final=step == 99)
                if reward is not None and strength:
                    # One cyclic interior site, canonical clean-posterior completion.
                    site = 1 + (step % length)
                    completed = q[:, 1:-1, 4:].argmax(-1).cpu().numpy()
                    sequences = ["".join(TOKENS[i] for i in row) for row in completed]
                    candidates = [
                        s[: site - 1] + aa + s[site:]
                        for s in sequences
                        for aa in TOKENS
                    ]
                    values = reward(candidates).reshape(count, 20)
                    utility = torch.zeros(
                        (count, 24), device=model.device, dtype=p.dtype
                    )
                    utility[:, 4:] = torch.as_tensor(
                        values, device=model.device, dtype=p.dtype
                    )
                    p[:, site] = guided_site(p[:, site], utility, strength)
                y = torch.multinomial(p.reshape(-1, 24), 1).reshape_as(x)
                if keep:
                    xs.append(x.cpu())
                    ys.append(y.cpu())
                    logs.append(log_probability(p, y).cpu())
                x = y
            for i, row in enumerate(x.cpu().tolist()):
                s = _decode_sample(i, tuple(row))
                rows.append(
                    {
                        **asdict(s),
                        "requested_length": length,
                        "draw_id": f"{label}-L{length}-{i}",
                    }
                )
            if keep:
                traces.append(
                    {
                        "length": length,
                        "x": torch.stack(xs),
                        "y": torch.stack(ys),
                        "logp": torch.stack(logs),
                    }
                )
    return rows, traces


def nft_update(policy, old, opt, rows, rewards, seed):
    policy.model.eval()
    old.model.eval()
    r = nft_optimality(rewards, [s["requested_length"] for s in rows])
    rng = random.Random(seed)
    order = []
    while len(order) < 6400:
        epoch = list(range(100))
        rng.shuffle(epoch)
        order.extend(epoch)
    losses = []
    with torch.random.fork_rng(devices=[policy.device]):
        torch.manual_seed(seed)
        for step in range(100):
            ids = order[step * 64 : (step + 1) * 64]
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for length in LENGTHS:
                ix = [i for i in ids if rows[i]["requested_length"] == length]
                if not ix:
                    continue
                x1 = policy._tokens([rows[i]["sequence"] for i in ix])
                x0 = torch.randint(0, 24, x1.shape, device=policy.device)
                t = torch.rand(len(ix), device=policy.device) * 0.999
                path = policy.path.sample(t=t, x_0=x0, x_1=x1)
                q = policy.model(path.x_t, path.t).softmax(-1)
                with torch.no_grad():
                    oldq = old.model(path.x_t, path.t).softmax(-1)
                target = torch.nn.functional.one_hot(x1, 24).to(q.dtype)
                loss = (
                    nft_loss(
                        q,
                        oldq,
                        target,
                        torch.as_tensor(r[ix], device=policy.device, dtype=q.dtype),
                    ).sum()
                    / 64
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("NFT loss")
                loss.backward()
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.1)
            if not torch.isfinite(norm):
                raise FloatingPointError("NFT gradients")
            opt.step()
            losses.append(total)
    return {"losses": losses, "optimality": r.tolist(), "optimizer_steps": 100}


def grpo_update(policy, base, opt, traces, rewards, seed):
    policy.model.eval()
    base.model.eval()
    offset = 0
    adv = []
    for tr in traces:
        n = tr["x"].shape[1]
        r = rewards[offset : offset + n]
        offset += n
        adv.append(
            torch.tensor(
                (r - r.mean()) / max(r.std(), 1e-8),
                device=policy.device,
                dtype=torch.float32,
            )
        )
    rng = np.random.default_rng(seed)
    losses = []
    kls = []
    clipped = []
    initial_errors = []
    # Exact unchanged-policy check including deterministic initial and terminal kernels.
    with torch.no_grad():
        for tr in traces:
            for step in (0, 30, 70, 99):
                x = tr["x"][step].to(policy.device)
                y = tr["y"][step].to(policy.device)
                t = torch.full((len(x),), step * 0.01, device=policy.device)
                p = transition(policy.model(x, t).softmax(-1), x, t, final=step == 99)
                err = (
                    (log_probability(p, y) - tr["logp"][step].to(policy.device))
                    .abs()
                    .max()
                    .item()
                )
                initial_errors.append(err)
        if max(initial_errors) > 1e-4:
            raise ValueError("on-policy ratio identity failed")
    for step in range(100):
        gi = int(rng.choice(3, p=[0.35, 0.40, 0.25]))
        tr = traces[gi]
        n = tr["x"].shape[1]
        ids = torch.tensor(rng.integers(0, n, 32), dtype=torch.long)
        # t=0 is deterministically unchanged and contributes zero gradient; omit it.
        ti = int(rng.integers(1, 100))
        x = tr["x"][ti, ids].to(policy.device)
        y = tr["y"][ti, ids].to(policy.device)
        t = torch.full((len(x),), ti * 0.01, device=policy.device)
        p = transition(policy.model(x, t).softmax(-1), x, t, final=ti == 99)
        with torch.no_grad():
            ref = transition(base.model(x, t).softmax(-1), x, t, final=ti == 99)
        logp = log_probability(p, y)
        oldlog = tr["logp"][ti, ids].to(policy.device)
        kl = (
            (p * (p.clamp_min(1e-30).log() - ref.clamp_min(1e-30).log()))
            .sum((-1, -2))
            .mean()
        )
        loss = ppo_loss(logp, oldlog, adv[gi][ids.to(policy.device)]) + 0.01 * kl
        if not torch.isfinite(loss):
            raise FloatingPointError("GRPO loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 0.1)
        if not torch.isfinite(norm):
            raise FloatingPointError("GRPO gradients")
        opt.step()
        losses.append(float(loss.detach()))
        kls.append(float(kl.detach()))
        clipped.append(
            float(((logp.detach() - oldlog).exp().sub(1).abs() > 0.2).float().mean())
        )
    return {
        "losses": losses,
        "base_kl": kls,
        "clip_fraction": clipped,
        "initial_logratio_error": max(initial_errors),
        "optimizer_steps": 100,
        "advantages": [a.cpu().tolist() for a in adv],
    }


def load_snapshot(root):
    root = Path(root)
    manifest = json.loads((root / "snapshot.json").read_text())
    if (
        manifest.get("protocol")
        not in ("cpp-interim-observer-v1", "cpp-baseline-observer-v1")
        or manifest.get("feedback_to_training") is not False
    ):
        raise ValueError("invalid interim snapshot protocol")
    cohorts_path = root / "evaluation/cohorts.json"
    receipt = json.loads((root / "evaluation/cohort_receipt.json").read_text())
    if receipt["snapshot_sha256"] != sha(root / "snapshot.json") or receipt[
        "cohorts_sha256"
    ] != sha(cohorts_path):
        raise ValueError("interim cohort binding mismatch")
    cohorts = json.loads(cohorts_path.read_text())
    names = [s["generator_id"] for s in manifest["checkpoints"]]
    if set(cohorts) != set(names):
        raise ValueError("interim checkpoint coverage differs")
    for name, rows in cohorts.items():
        if len(rows) != 100 or len({r["candidate_id"] for r in rows}) != 100:
            raise ValueError("interim cohort count or identity differs")
        if {k: sum(r["length"] == k for r in rows) for k in (12, 24, 38)} != {
            12: 35,
            24: 40,
            38: 25,
        }:
            raise ValueError("interim length mixture differs")
        if any(len(r["sequence"]) != r["length"] for r in rows):
            raise ValueError("peptide length metadata differs")
    return {name: cohorts[name] for name in names}


def endpoint(root, model, base, arm, reward):
    path = root / "evaluation/cohorts.json"
    if path.exists():
        write(
            root / "evaluation/cohort_receipt.json",
            {
                "snapshot_sha256": sha(root / "snapshot.json"),
                "cohorts_sha256": sha(path),
            },
        )
        load_snapshot(root)
        return
    cohorts = {}
    for name in ("p0", arm):
        current = base if name == "p0" else model
        if name == arm and arm == "guidance":
            rows, _ = discrete_draw(current, SEED + 900000, name, reward, 5.0)
        else:
            rows = vanilla_draw(current, SEED + 900000, name)
        write(root / "evaluation" / f"{name}-raw.json", rows)
        validate_rows(rows)
        cohorts[name] = [
            {
                "candidate_id": f"eval-l{r['requested_length']}-{i:04d}",
                "sequence": r["sequence"],
                "length": r["requested_length"],
            }
            for i, r in enumerate(rows)
        ]
        write(
            root / "evaluation" / f"{name}-reward.json",
            {"reward": reward([r["sequence"] for r in rows]).tolist()},
        )
    write(
        root / "snapshot.json",
        {
            "protocol": "cpp-baseline-observer-v1",
            "feedback_to_training": False,
            "checkpoints": [{"generator_id": name} for name in cohorts],
            "seed": SEED,
            "checkpoint_sha256": sha(root / "final.pt"),
            "baseline": arm,
            "sampling": "released PepDFM sampler except guidance uses distribution-equivalent marginalized transition with fixed reward tilt",
        },
    )
    write(path, cohorts)
    write(
        root / "evaluation/cohort_receipt.json",
        {"snapshot_sha256": sha(root / "snapshot.json"), "cohorts_sha256": sha(path)},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method", choices=["dpo", "nft", "grpo", "guidance"], required=True
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--cellppd-source", type=Path, required=True)
    args = p.parse_args()
    args.arm = {"nft": "nft7", "grpo": "flowgrpo5"}.get(args.method, args.method)
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("SLURM_JOB_ID"):
        raise PermissionError("tracked allocation required")
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = Path(__file__).with_name("configs") / "baselines.json"
    if (
        sha(config_path)
        != "a6d3f2b0af171fe5d1f34f1b9c67874a5f8337fc04c787dbb94793216ccf17c2"
    ):
        raise ValueError("baseline protocol hash differs")
    config = read(config_path)[args.method]
    write(root / "protocol.json", config)
    attempt = root / "attempts" / os.environ["SLURM_JOB_ID"]
    attempt.mkdir(parents=True, exist_ok=True)
    write(
        attempt / "environment.json",
        {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "git": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "packages": subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True
            ),
            "job_id": os.environ["SLURM_JOB_ID"],
        },
    )
    from domains.peptides.model import PepDfmAssets, PEPDFM_TOKEN_BY_ID
    from domains.peptides.checkpoint import PepDfmCheckpoint

    if TOKENS != "".join(PEPDFM_TOKEN_BY_ID[4:]):
        raise ValueError("token order mismatch")
    import joblib

    source = args.cellppd_source.resolve()
    if (root / "reward.pkl").exists() and not (root / "reward_receipt.json").exists():
        (root / "reward.pkl").rename(attempt / "orphan_reward.pkl")
        if (root / "reward_validation.json").exists():
            (root / "reward_validation.json").rename(
                attempt / "orphan_reward_validation.json"
            )
    if not (root / "reward.pkl").exists():
        rf, validation = fit_reward(source)
        write(root / "reward_validation.json", validation)
        joblib.dump(rf, root / "reward.pkl")
        write(root / "reward_receipt.json", {"sha256": sha(root / "reward.pkl")})
    if sha(root / "reward.pkl") != read(root / "reward_receipt.json")["sha256"]:
        raise ValueError("RF hash mismatch")
    reward = Reward(joblib.load(root / "reward.pkl"))
    print("Reward validated", flush=True)
    base = PepDfmCheckpoint(PepDfmAssets.resolve(args.assets), device="cuda:0")
    policy = base.clone()
    started = time.monotonic()
    if (root / "final.pt").exists() and not (root / "training_complete.json").exists():
        (root / "final.pt").rename(attempt / "orphan_final.pt")
    if (root / "training_complete.json").exists():
        meta = read(root / "training_complete.json")
        policy.load_checkpoint(
            root / "final.pt", expected_sha256=meta["checkpoint_sha256"]
        )
    elif args.arm == "dpo":
        from domains.peptides.dpo import (
            FlowDPOConfig,
            fine_tune_flow_dpo,
            write_or_validate_pair_manifest,
        )

        rows = vanilla_draw(base, SEED + 1000000, "train")
        write(root / "training_draws.json", rows)
        validate_rows(rows)
        scores = reward([r["sequence"] for r in rows])
        unique = {r["sequence"]: float(v) for r, v in zip(rows, scores)}
        ranked = sorted(unique, key=lambda s: (unique[s], s))
        if len(ranked) < 40 or unique[ranked[19]] >= unique[ranked[-20]]:
            raise ValueError("no strict20x20 preferences")
        pos, neg = ranked[-20:], ranked[:20]
        write(
            root / "selection.json",
            {
                "positive": pos,
                "negative": neg,
                "all_rewards": unique,
                "cutoff_ties": "lexical tie-break within a side; all chosen must strictly exceed all rejected",
            },
        )
        pairs = root / "pairs.jsonl"
        write_or_validate_pair_manifest(
            pairs, positive_sequences=pos, negative_sequences=neg
        )
        trace = fine_tune_flow_dpo(
            policy=policy,
            reference=base,
            positive_sequences=pos,
            negative_sequences=neg,
            pair_manifest_path=pairs,
            config=FlowDPOConfig(
                learning_rate=5e-6,
                steps=100,
                batch_size=32,
                beta=2.6,
                gradient_clip_norm=0.1,
                seed=SEED + 3000000,
            ),
        )
        write(root / "training_trace.json", trace)
    elif args.arm in (*NFT_ARMS, "flowgrpo5"):
        old = base.clone()
        for param in old.model.parameters():
            param.requires_grad_(False)
        opt = torch.optim.Adam(
            policy.model.parameters(), lr=1e-5 if args.arm in NFT_ARMS else 5e-6
        )
        for iteration in range(1, update_count(args.arm) + 1):
            directory = root / f"round-{iteration:02d}"
            receipt = directory / "complete.json"
            if receipt.exists():
                meta = read(receipt)
                if sha(directory / "state.pt") != meta["state_sha256"]:
                    raise ValueError("resume hash mismatch")
                state = torch.load(
                    directory / "state.pt",
                    map_location=policy.device,
                    weights_only=True,
                )
                policy.model.load_state_dict(state["policy"])
                old.model.load_state_dict(state["old"])
                opt.load_state_dict(state["optimizer"])
                continue
            if directory.exists():
                directory.rename(attempt / f"orphan_round-{iteration:02d}")
            seed = SEED + 1000000 + (iteration - 1) * 100000
            if args.arm in NFT_ARMS:
                rows = vanilla_draw(old, seed, f"round{iteration}")
                traces = None
            else:
                rows, traces = discrete_draw(
                    policy, seed, f"round{iteration}", keep=True
                )
            write(directory / "rollouts.json", rows)
            validate_rows(rows)
            scores = reward([r["sequence"] for r in rows])
            write(directory / "rewards.json", scores.tolist())
            if args.arm in NFT_ARMS:
                trace = nft_update(
                    policy, old, opt, rows, scores, SEED + 4000000 + iteration * 100000
                )
                eta = min(0.001 * iteration, 0.5)
                with torch.no_grad():
                    for a, b in zip(old.model.parameters(), policy.model.parameters()):
                        a.mul_(eta).add_(b, alpha=1 - eta)
            else:
                torch.save(traces, directory / "transitions.pt")
                trace = grpo_update(
                    policy,
                    base,
                    opt,
                    traces,
                    scores,
                    SEED + 5000000 + iteration * 100000,
                )
            write(directory / "training.json", trace)
            torch.save(
                {
                    "policy": policy.model.state_dict(),
                    "old": old.model.state_dict(),
                    "optimizer": opt.state_dict(),
                },
                directory / "state.pt",
            )
            digest = policy.save_checkpoint(directory / "student.pt")
            write(
                receipt,
                {
                    "state_sha256": sha(directory / "state.pt"),
                    "checkpoint_sha256": digest,
                    "update": iteration,
                },
            )
            print(
                args.arm,
                "update",
                iteration,
                "mean training reward",
                float(scores.mean()),
                flush=True,
            )
    if not (root / "training_complete.json").exists():
        digest = policy.save_checkpoint(root / "final.pt")
        write(
            root / "training_complete.json",
            {
                "arm": args.arm,
                "checkpoint_sha256": digest,
                "reward_sha256": sha(root / "reward.pkl"),
                "reward_calls_this_attempt": reward.calls,
                "optimizer_steps": 100 * update_count(args.arm),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    endpoint(root, policy, base, args.arm, reward)
    write(
        attempt / "finished.json",
        {
            "reward_calls_this_attempt": reward.calls,
            "arm": args.arm,
            "training_and_generation_complete": True,
        },
    )
    print("Training and endpoint samples complete", args.arm, flush=True)


if __name__ == "__main__":
    main()
