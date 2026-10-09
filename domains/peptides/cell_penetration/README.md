# Mammalian cell penetration

This task starts from pretrained PepDFM. The LLM constructs a curriculum from
the current generator's candidates, supplies categorical judgments, and uses
feedback from successive DPO updates to refine the intermediate goals.

## Run

Complete the root [installation and API setup](../../../README.md#installation)
and [peptide runtime setup](../../../README.md#peptide-runtime). From the source
checkout, inside a Slurm GPU allocation:

```bash
llm-to-flow run cell-penetration \
  --assets /path/to/assets/mog-dfm \
  --output-dir runs/cell-penetration
```

The command prepares the task, checks its assets and runtime, generates an
initial pool, and runs the adaptive curriculum. The sampling-length mixture
stays fixed while goals and generator checkpoints evolve.

See [LLM traces and output folders](../../../README.md#llm-traces-and-output-folders)
for the files created automatically under your output directory.

## Adapt the task

- [configs/dpo.json](configs/dpo.json): models, API budgets, curriculum controls,
  and DPO settings.
- [configs/pool.json](configs/pool.json) and
  [configs/length_policy.json](configs/length_policy.json): initial sampling.
- [../prompts.py](../prompts.py): `CELL_PENETRATION` teacher prompts.

Update the task's goal, prompts, and validation rules together, commit the
changes, and launch a fresh run. Use `--runtime-python` if the GPU dependencies
live in a separate environment.

## Optional baselines

[baselines.py](baselines.py) provides DPO, DiffusionNFT, FlowGRPO, and guidance;
[configs/baselines.json](configs/baselines.json) contains their settings.
The baseline runner also requires the upstream CellPPD source:

```bash
git clone https://github.com/raghavagps/CellPPD.git /path/to/CellPPD
git -C /path/to/CellPPD checkout 4708e09d5a28cc9d843b3bf70c7d1ec93e79d881
python -m domains.peptides.cell_penetration.baselines --method nft \
  --root runs/cpp-nft --assets /path/to/assets/mog-dfm \
  --cellppd-source /path/to/CellPPD
```

Run baselines in a GPU allocation too. Downstream evaluation is separate from
the curriculum; its asset requirements are described in
[evaluation.json](evaluation.json).
