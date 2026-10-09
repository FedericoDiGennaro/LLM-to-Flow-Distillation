# Peptides

Both peptide tasks use **PepDFM**. The LLM proposes intermediate goals and
judges generated candidates; its preferences train the generator with DPO.
Each update starts from the preceding generator checkpoint.

| Task | Guide |
| --- | --- |
| Selective anticancer activity | [anticancer/README.md](anticancer/README.md) |
| Mammalian cell penetration | [cell_penetration/README.md](cell_penetration/README.md) |

Follow the root [installation and API setup](../../README.md#installation) and
[peptide runtime setup](../../README.md#peptide-runtime), then use the command
in the task guide.

## Prompts and configuration

All prompt text is in [prompts.py](prompts.py). `ANTICANCER` and
`CELL_PENETRATION` contain the prompts used by the corresponding fresh runs.

| Key | Purpose |
| --- | --- |
| `LLM-Judge` | Judge candidates against the current goal. |
| `SetGoal_proposer` | Propose the next intermediate goal. |
| `goal_controller` | Correct a goal proposal using generator feedback. |

Each task's `configs/dpo.json` defines the models, budgets, curriculum controls,
and DPO settings. The model and training implementation is shared in
[model.py](model.py), [checkpoint.py](checkpoint.py), and [dpo.py](dpo.py).
To adapt a task, update its configuration, prompts, and matching validation
rules, commit the changes, and launch a fresh run.
