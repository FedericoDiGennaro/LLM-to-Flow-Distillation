![GSK3β docking visualization supplied with the project](result.png)

# GSK3β binding and aqueous solubility

This task adapts pretrained **FlowMol3**. The LLM judges generated candidates,
proposes reachable intermediate goals, and supplies preferences for sequential
DPO updates. The curriculum is built from the generator's evolving outputs.

## Setup and execution

Complete the root [installation and API setup](../../../README.md#installation).
From the source checkout, install the molecular runtime with Pixi and fetch
FlowMol3:

```bash
python -m domains.molecules.gsk3b.setup --ltfd-home /path/to/flowmol-runtime
```

Inside a Slurm GPU allocation, start a fresh run:

```bash
llm-to-flow run gsk3b \
  --runtime-home /path/to/flowmol-runtime \
  --output-dir runs/gsk3b
```

The launcher reads the setup receipt to locate the GPU interpreter, source,
checkpoint, and support files. Add `--smoke` to use the smaller integration
profile. Setup targets Linux x86-64 with the CUDA environment in
[../pixi.toml](../pixi.toml).

See [LLM traces and output folders](../../../README.md#llm-traces-and-output-folders)
for the files created automatically under your output directory.

## Adapt the task

- [configs/dpo.json](configs/dpo.json): models, API budgets, curriculum limits,
  sampling, and DPO settings.
- [../prompts.py](../prompts.py): the `GSK3B` teacher prompts.
- [cards.py](cards.py): candidate evidence and the task goal.
- [baselines/](baselines/): optional DPO, DiffusionNFT, FlowGRPO, guidance,
  and reward-tilting comparisons.

Update the goal in the configuration, prompts, candidate cards, and validation
rules together. Commit changes before launching a fresh run. The adaptive
training loop saves generated candidates and checkpoints; docking and other
external evaluations are separate workflows.

## Baseline and reproduction boundaries

The optional baseline campaign profiles are `baselines/dpo.json`, `nft.json`,
`grpo.json`, `guidance.json`, and `tilt.json`. Their optimization reward is the
TDC **GSK3B random-forest probability**.
They are single-reward baselines; they do not optimize the LLM's full joint
binding-and-solubility goal. A fair final comparison needs the same separately
frozen binding, solubility, validity, and sampling-budget evaluation across arms.
Improvement in the training reward is a development metric, not that comparison.

Flow-DPO uses a flow-matching preference surrogate, not exact molecular likelihoods.
The FlowGRPO-style adapter uses a flow-loss log-density surrogate and a KL proxy;
it is not a literal reproduction of continuous-image Flow-GRPO. The NFT adapter
uses FlowMol's coordinate and categorical losses. Label these adaptations as such
when reporting them. Reward tilting also uses more proposals than the other arms;
report that budget difference rather than treating it as matched sampling.

The resolved campaign profile owns the training schedule. The standalone file
`configs/preference_training/flowmol_diffusion_nft_v1.json` records adapter
defaults, not the active multi-round NFT schedule. `baselines/table.json` is a
historical provenance record: its hashes identify checkpoints but do not supply
them. Reproducing that table additionally requires those checkpoint files, saved
judgments/configurations, evaluation inputs, and evaluation outputs. A new run
with the shipped defaults is not guaranteed to recover the same LLM curriculum
or table, even with matching sampling seeds. The table's five sampling seeds
are not five independently trained models.

Resume with the original source revision and runtime artifacts. Failed saved
non-collapse receipts remain failures; they cannot be bypassed by restarting.
Every subsequent training update rechecks label support, repeat agreement, and
abstention. Provider revisions are restored from saved responses before new calls.
This review version tightens artifact checks; it is not a drop-in authorization
to resume old runs. Older NFT reward files without a rollout-content hash require
a separate verified recovery, not an invented hash or overwritten receipt.
