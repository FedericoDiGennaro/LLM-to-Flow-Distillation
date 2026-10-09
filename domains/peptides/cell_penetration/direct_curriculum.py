"""Small controller/auditor repair; exact-F judge and optimizer are unchanged."""
from __future__ import annotations

from domains.peptides.prompts import CELL_PENETRATION as DOMAIN_PROMPTS

import re
from typing import Mapping, Any

from pydantic import Field

from domains.peptides.cell_penetration.teacher import MovingFrontierSemanticAuditResponse


DIRECT_CONTROLLER_ADDENDUM = DOMAIN_PROMPTS['direct_controller_addendum']

DIRECT_AUDITOR_ADDENDUM = DOMAIN_PROMPTS['direct_auditor_addendum']


class DirectGoalAuditResponse(MovingFrontierSemanticAuditResponse):
    operative_biological_capability: bool
    sequence_reasoning_beyond_proxy_lookup: bool
    no_unjustified_necessary_proxy: bool
    final_positive_counterexample_check: str = Field(min_length=1)


DIRECT_AUDIT_FLAGS = (
    'operative_biological_capability',
    'sequence_reasoning_beyond_proxy_lookup',
    'no_unjustified_necessary_proxy',
)


def operative_contract_failures(goal: Mapping[str, Any]) -> list[str]:
    """Catch the observed literal proxy-rule escape independently of the auditor.

    Only operative text is checked: describing a band in limitations/bridge is
    allowed. Biological containment still requires the semantic counterexample
    audit; these narrow checks do not pretend to establish it automatically.
    """
    text=' '.join(str(goal.get(k,'')) for k in (
        'frontier_criterion','positive_evidence_definition','negative_evidence_definition'))
    text=text.lower().replace('–','-').replace('—','-')
    failures=[]
    if re.search(r'\b(?:lower|middle|upper)[\s-]*(?:third|quartile|quintile|decile)|'
                 r'\b(?:tertile|quantile|percentile)s?\b|\b(?:top|bottom)\s+\d+\s*%',text):
        failures.append('operative_predictor_quantile_rule')
    if re.search(r'\b(?:cysteine|cys)[\s-]*free\b|'
                 r'\b(?:lack\w*|without|no|exclude\w*|avoid\w*)\s+(?:any\s+)?(?:cysteine|cys)\b|'
                 r'\bfree\s+of\s+(?:cysteine|cys)\b',text):
        failures.append('unjustified_cysteine_exclusion')
    return failures


def direct_contract_result(semantic: Mapping[str, Any], goal: Mapping[str, Any]) -> dict[str, Any]:
    failures=operative_contract_failures(goal)
    failures.extend('audit_'+field for field in DIRECT_AUDIT_FLAGS if semantic.get(field) is not True)
    return {**semantic,'accepted':semantic.get('accepted') is True and not failures,
        'direct_curriculum_contract_version':'biological-capability-v1',
        'direct_contract_failed_conditions':failures}
