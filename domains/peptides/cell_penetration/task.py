"""Explicit process-local task binding; no mutable shared arm configuration."""
from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class PeptideTask:
    task_id: str
    experiment_id: str
    seed: int
    final_goal: str
    activity: str
    interpretation: str
    reward: str
    evaluator: str

TASKS = {"cell_penetration": PeptideTask(
        'cell_penetration', 'EXP-0186', 18601,
        'Generate peptides with promising mammalian-cell-penetrating activity while minimizing toxicity to mammalian cells.',
        'mammalian-cell-penetrating activity',
        'Judge plausible cellular uptake rather than mere surface binding or membrane destruction. '
        'Do not prescribe a cationic motif, receptor or uptake route. '
        'Cell penetration alone does not establish blood-brain-barrier crossing, '
        'cytosolic escape or delivery of an arbitrary cargo. No measured uptake is supplied.',
        'Frozen AAC+DPC random forest on CellPPD CPP/non-CPP training labels',
        'pLM4CPPs ESM2-640 CNN with frozen training-time normalization')}

def task_profile(task_id: str) -> PeptideTask:
    if task_id not in TASKS:
        raise ValueError('PEPDFM_TASK_ID must be cell_penetration')
    return TASKS[task_id]

TASK = task_profile(os.environ.get('PEPDFM_TASK_ID', 'cell_penetration'))
