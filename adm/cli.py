"""Run the bundled adaptive curricula from a clean source checkout."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from adm.preferences.provider import provider_key, provider_settings, without_credentials

TASKS = {
    "anticancer": "domains.peptides.anticancer",
    "cell-penetration": "domains.peptides.cell_penetration",
    "gsk3b": "domains.molecules.gsk3b",
    "drd2": "domains.molecules.drd2",
    "kinase": "domains.molecules.kinase",
}


MOLECULAR_TASKS = {task for task, module in TASKS.items() if module.startswith("domains.molecules.")}


def checkout_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if result.returncode:
        raise ValueError("run from a clone of LLM-to-Flow-Distillation; see README.md#running-experiments")
    root = Path(result.stdout.strip()).resolve()
    if not all((root / module.replace(".", "/") / "run.py").is_file() for module in TASKS.values()):
        raise ValueError("current checkout is not LLM-to-Flow-Distillation")
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if dirty.strip():
        raise ValueError("commit your changes before starting a fresh run")
    return root


def module_command(python: Path, module: str, **options) -> list[str]:
    command = [str(python), "-m", module]
    for name, value in options.items():
        if value is None or value is False:
            continue
        command.append("--" + name.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    return command


def preparation(args, root: Path, python: Path) -> list[str]:
    module = TASKS[args.task]
    if args.task == "cell-penetration":
        deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=args.valid_days)
        return module_command(python, module + ".prepare", output_root=args.output_dir,
                              deadline_utc=deadline.isoformat())
    options = dict(output_root=args.output_dir, out_dir=args.output_dir / "control",
                   valid_days=args.valid_days)
    if args.task in MOLECULAR_TASKS:
        options.update(mode="smoke" if args.smoke else "run",
                       config=root / module.replace(".", "/") / "configs" /
                       ("smoke.json" if args.smoke else "dpo.json"))
    return module_command(python, module + ".prepare", **options)


def peptide_steps(args, root: Path, python: Path, runtime: Path):
    module = TASKS[args.task]
    output = args.output_dir
    cpp = args.task == "cell-penetration"
    config = output / "template.json" if cpp else root / module.replace(".", "/") / "configs/dpo.json"
    pool = output / "pool_config.json" if cpp else root / module.replace(".", "/") / "configs/pool.json"
    policy = output / ("length_policy.json" if cpp else "campaign_length_policy.json")
    control = output if cpp else output / "control"
    authorization = control / ("authorization.json" if cpp else "gemini_authorization.json")
    campaign = control / "ACTIVE_CAMPAIGN.json"
    # A normal peptide installation already has torchdiffeq on sys.path. The
    # optional overlay remains available for a separate GPU environment.
    overlay = args.dependency_overlay or root
    steps = [
        (False, module_command(runtime, "domains.peptides.fetch_assets", asset_dir=args.assets, local_files_only=True)),
        (False, [str(runtime), "-c", "import torch, torchdiffeq, xgboost; assert torch.cuda.is_available(), 'CUDA GPU unavailable'; assert xgboost.__version__ == '2.1.4', 'install xgboost==2.1.4'"]),
        (False, module_command(runtime, "domains.peptides.check_predictors", asset_dir=args.assets,
                               output=output / "runtime/predictors.json")),
    ]
    if not cpp:
        steps.extend([
            (True, module_command(python, "domains.peptides.preflight", config=config,
                                  output_root=output / "campaign", pool_config=pool,
                                  authorization=authorization, active_campaign=campaign)),
            (True, module_command(python, "domains.peptides.select_lengths", config=config,
                                  artifact=policy, output_root=output / "campaign",
                                  authorization=authorization, active_campaign=campaign)),
        ])
    steps.extend([
        (False, module_command(runtime, module + ".pool", config=pool, length_policy=policy,
                               output_root=output / "p0", asset_dir=args.assets, active_campaign=campaign)),
        (False, module_command(python, module + ".freeze", template=config,
                               candidates=output / "p0/candidate_pool.jsonl",
                               evidence_reference=output / "p0/frozen_evidence_reference_private.json",
                               length_policy=policy, output=output / "frozen_config.json")),
        (True, module_command(python, module + ".run", config=output / "frozen_config.json",
                              output_root=output / "campaign", asset_dir=args.assets,
                              authorization=authorization, active_campaign=campaign,
                              pepdfm_python=runtime, dependency_overlay=overlay)),
    ])
    return steps


def molecular_steps(args, root: Path, python: Path):
    receipt_path = args.runtime_home / "assets/flowmol3/setup_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    command = module_command(
        python, TASKS[args.task] + ".run",
        config=root / TASKS[args.task].replace(".", "/") / "configs" / ("smoke.json" if args.smoke else "dpo.json"),
        output_root=args.output_dir, checkpoint=receipt["checkpoint"],
        n_atoms_hist=receipt["n_atoms_hist_file"], marginal_dists=receipt["marginal_dists_file"],
        flowmol_python=receipt["runtime_python"],
        authorization=args.output_dir / "control/gemini_authorization.json",
        active_campaign=args.output_dir / "control/ACTIVE_CAMPAIGN.json",
    )
    return [(True, command)], Path(receipt["source"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="start a fresh adaptive curriculum")
    run.add_argument("task", choices=TASKS)
    run.add_argument("--output-dir", type=Path, required=True, help="new or empty directory for traces, candidates, and checkpoints")
    run.add_argument("--assets", type=Path, help="downloaded PepDFM asset directory")
    run.add_argument("--runtime-python", type=Path, help="optional separate peptide GPU interpreter")
    run.add_argument("--dependency-overlay", type=Path, help="optional peptide dependency overlay")
    run.add_argument("--runtime-home", type=Path, default=Path(".cache/flowmol3"), help="FlowMol setup directory")
    run.add_argument("--smoke", action="store_true", help="use the small molecular profile")
    run.add_argument("--valid-days", type=int, default=7, help="run deadline in days")
    run.add_argument("--prepare-only", action="store_true", help="prepare inputs without GPU work or API calls")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = checkout_root()
        args.output_dir = args.output_dir.resolve()
        if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
            raise ValueError("choose a new or empty output directory")
        if args.valid_days <= 0:
            raise ValueError("--valid-days must be positive")
        if args.smoke and args.task not in MOLECULAR_TASKS:
            raise ValueError("--smoke is available for molecular tasks")
        python = Path(sys.executable).absolute()
        runtime = args.runtime_python.absolute() if args.runtime_python else python
        args.runtime_home = args.runtime_home.resolve()
        if args.assets:
            args.assets = args.assets.resolve()
        if args.dependency_overlay:
            args.dependency_overlay = args.dependency_overlay.resolve()
        environment = dict(os.environ)
        paths = [str(root)]
        if args.dependency_overlay:
            paths.append(str(args.dependency_overlay))
        if environment.get("PYTHONPATH"):
            paths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(paths)
        environment["PEPDFM_E2E_OUTPUT_ROOT"] = str(args.output_dir)
        environment["ACTFLOW_CPP_OUTPUT_ROOT"] = str(args.output_dir)
        environment["MOLECULES_E2E_OUTPUT_ROOT"] = str(args.output_dir)
        environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        environment.setdefault("PYTHONHASHSEED", "0")
        if args.task == "cell-penetration":
            environment.update(PEPDFM_TASK_ID="cell_penetration", PEPDFM_RUN_EXPERIMENT_ID="EXP-0189")
        steps = []
        if not args.prepare_only:
            if not environment.get("SLURM_JOB_ID", "").isdigit():
                raise ValueError("run inside an allocated Slurm GPU job")
            provider_settings()
            provider_key()
            if args.task in MOLECULAR_TASKS:
                steps, source = molecular_steps(args, root, python)
                environment["PYTHONPATH"] = str(source) + os.pathsep + environment["PYTHONPATH"]
            else:
                if not args.assets or not args.assets.is_dir():
                    raise ValueError("--assets must point to downloaded PepDFM assets")
                steps = peptide_steps(args, root, python, runtime)
        commands = [(False, preparation(args, root, python)), *steps]
        for needs_api, command in commands:
            child_environment = dict(environment)
            if not needs_api:
                child_environment = without_credentials(child_environment)
            print("+ " + shlex.join(command), flush=True)
            subprocess.run(command, cwd=root, env=child_environment, check=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
