# Selective anticancer peptides

Start from pretrained PepDFM and adapt it through intermediate goals proposed
by the LLM. Candidate judgments form preference pairs for each DPO update;
feedback from the updated generator determines the next curriculum step.

## Run

Complete the root [installation and API setup](../../../README.md#installation)
and [peptide runtime setup](../../../README.md#peptide-runtime). From the source
checkout, inside a Slurm GPU allocation:

```bash
llm-to-flow run anticancer \
  --assets /path/to/assets/mog-dfm \
  --output-dir runs/anticancer
```

The command prepares the run, checks the assets and predictor runtime, selects
an initial sampling-length policy, generates the initial pool, and starts the
adaptive goal/DPO loop. Outputs go to the chosen directory.

See [LLM traces and output folders](../../../README.md#llm-traces-and-output-folders)
for the files created automatically under your output directory.

## Adapt the task

- [configs/dpo.json](configs/dpo.json): models, API budgets, curriculum limits,
  and DPO settings.
- [configs/pool.json](configs/pool.json): initial sampling settings.
- [../prompts.py](../prompts.py): `ANTICANCER` teacher prompts.
- [baselines/README.md](baselines/README.md): optional reward-based comparisons.

Change the task goal, prompts, and corresponding validation rules together.
Commit edits before starting a fresh run. A separate GPU environment can be
selected with `--runtime-python /path/to/peptide-env/bin/python`; both
interpreters must have their required dependencies installed.
