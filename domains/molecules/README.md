# Molecules

The three molecular tasks use **FlowMol3**. An LLM proposes intermediate goals
and judges candidate evidence; its preferences update the generator with DPO.

Follow the root [installation and API setup](../../README.md#installation).
One [FlowMol3 runtime](pixi.toml) and set of pretrained assets serve all three
tasks. Each task keeps its own prompts, configuration, and evaluation protocol.

| Task | Command | Task guide | Comparison reward |
| --- | --- | --- | --- |
| Dopamine D2 receptor binding with moderate lipophilicity | `llm-to-flow run drd2` | [DRD2](drd2/README.md) | TDC/Olivecrona DRD2 probability × a MolLogP [1, 3] indicator |
| GSK3β binding and aqueous solubility | `llm-to-flow run gsk3b` | [GSK3β](gsk3b/README.md) | TDC GSK3β probability |
| Kinase-focused design with specified ring motifs | `llm-to-flow run kinase` | [Kinase design](kinase/README.md) | Binary graph verifier for the requested structural motifs |

Add `--runtime-home /path/to/flowmol-runtime --output-dir /path/to/new-run`.
`--prepare-only` writes inputs without GPU work or API calls. `--smoke` selects
a smaller profile that still uses the GPU and API. Both Gemini and the optional
LiteLLM Responses gateway are supported through the root API setup.

The kinase task concerns structural requirements, rather than predicted binding
to a particular kinase: an amino-substituted six-membered aza-aromatic ring and
a distinct saturated ring with a free secondary amine. Its task guide gives the
full goal and graph-verification criteria.

## Prompts and configuration

| Task | Prompt text and request construction | Main configuration |
| --- | --- | --- |
| DRD2 | [drd2/teacher.py](drd2/teacher.py), [drd2/cards.py](drd2/cards.py) | [drd2/configs/dpo.json](drd2/configs/dpo.json) |
| GSK3β | `GSK3B` in [prompts.py](prompts.py), [gsk3b/teacher.py](gsk3b/teacher.py) | [gsk3b/configs/dpo.json](gsk3b/configs/dpo.json) |
| Kinase design | [kinase/teacher.py](kinase/teacher.py), [kinase/cards.py](kinase/cards.py) | [kinase/configs/dpo.json](kinase/configs/dpo.json) |

The teacher roles judge candidates, propose intermediate goals, correct proposals
using generator feedback, and audit goal transitions. Two independent judgment
passes yield stable positive/negative examples for training; abstentions and
inconsistent labels are excluded. Docking scores and scalar baseline rewards
are not shown to the teacher.

Configurations specify models, budgets, curriculum gates, sampling, and DPO
settings. The new DRD2 and kinase tasks use fresh working cohorts separately
from their development cohorts. Their task-specific advancement rules are
preserved: DRD2 requires both an absolute positive fraction and improvement;
kinase design uses a strict positive-fraction threshold. Refer to each guide
for its exact settings and baseline commands. When adapting a task, update its
goal, prompts, candidate cards, and validation rules together.
