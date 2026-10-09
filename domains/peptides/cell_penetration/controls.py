"""Parallel PepDFM task transfer of EXP184 curriculum and durable recovery.

No ACP imports or files are used by any training/control function. Scalar
diagnostics run in a separate key-free subprocess after terminal checkpointing.
"""
from __future__ import annotations

from domains.peptides.prompts import CELL_PENETRATION as DOMAIN_PROMPTS

import argparse
from collections import Counter
from dataclasses import replace
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
from domains.peptides.cell_penetration.task import TASK
from domains.peptides.cell_penetration.heuristics import ARMS as HEURISTIC_ARMS, policy_for_arm
ARMS = {arm: (True, True, 1e-5) for arm in HEURISTIC_ARMS}
SEEDS = (TASK.seed,)
SURROGATE_ONLY_ADDENDUM = DOMAIN_PROMPTS['surrogate_only_addendum']
CONTROLLER_ADDENDUM = DOMAIN_PROMPTS['controller_addendum']


def link_final_feedback(feedback, current_map):
    """Bind every F diagnosis to the SAME visible item ID as its sequence."""
    real_to_item={sample.candidate_id:item for item,sample in current_map.items()}
    passes=[]
    for key in ('current_final','current_final_repeat'):
        rows=feedback[key]['judgments']
        indexed={row['candidate_id']:row for row in rows}
        if len(indexed)!=len(rows) or set(indexed)!=set(real_to_item):
            raise ValueError('F-feedback candidate coverage differs from current cohort')
        passes.append(indexed)
    return {'scope':'current_working_cohort; item IDs match the sequences in that cohort',
        'stable_final_positive_fraction':feedback['stable_final_positive_fraction'],
        'entry_stable_final_positive_fraction':feedback['entry_stable_final_positive_fraction'],
        'by_item':[{'item_id':real_to_item[real],
            'pass_1':{k:v for k,v in passes[0][real].items() if k!='candidate_id'},
            'pass_2':{k:v for k,v in passes[1][real].items() if k!='candidate_id'}}
            for real in sorted(real_to_item,key=real_to_item.get)]}

def canonical_cell_key(judge, goal, goal_id, samples):
    """Scientific inputs determine one fixed two-pass assessment per cohort.

    Observed model revision is enforced by the durable model binding, separate
    from this key so the first observed version cannot change the cache key.
    Working and held-out roots are disjoint and cannot share this cache.
    """
    from adm.preferences.clients import canonical_json
    value={'protocol':'EXP110-fixed-two-pass-cell-v1','goal_id':goal_id,
        'final_goal':goal.final_goal,'frontier_criterion':goal.frontier_criterion,
        'operative_proxy_specification':goal.operative_proxy_specification(),
        'samples':[{'candidate_id':s.candidate_id,'sequence':s.sequence,
                    'developability_evidence':dict(s.developability_evidence)} for s in samples],
        'requested_model':judge.model,'reasoning_effort':judge.reasoning_effort,
        'system_prompt':judge.system_prompt,'experiment_context':judge.experiment_context,
        'chunk_size':judge.judge_chunk_size,'provider_order_seed':judge.provider_order_seed,
        'max_output_tokens':judge.max_output_tokens,
        'developability_evidence_tool':judge.developability_evidence_tool,
        'schema_source_sha256':digest(ROOT/'domains/peptides/cell_penetration/judgments.py'),
        'show_final_goal_context_for_intermediate':judge.show_final_goal_context_for_intermediate,
        'use_generic_evidence_cards':judge.use_generic_evidence_cards}
    return 'fixed-cell-'+hashlib.sha256(canonical_json(value).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def validate_arm(config):
    m = config['improvement_matrix']
    if m['arm'] not in ARMS or int(m['seed']) not in SEEDS:
        raise ValueError('unregistered matrix arm/seed')
    expected = ARMS[m['arm']]
    if m.get('final_updates', 0) != 0:
        raise PermissionError('direct F optimization is forbidden')
    if (m['extended_curriculum'], m['controller_feedback'], config['fine_tuning']['learning_rate']) != expected:
        raise ValueError('matrix intervention drift')
    for field,expected_flag in [('repaired_feedback',True),
                                ('canonical_goal_cells',True),
                                ('reasoning_prompt',False),
                                ('mechanism_neutral_guidance',True)]:
        if m.get(field) is not expected_flag: raise ValueError('registered repair/scaffold flag differs')
    if config.get('heuristic_control') != policy_for_arm(m['arm']):
        raise ValueError('registered heuristic policy drift')
    saturation = config.get('saturation_triggered_tightening')
    if saturation is not None and saturation != {
        'enabled': True, 'minimum_positive_fraction': .75,
        'minimum_gain_over_entry': .05, 'minimum_stable_positive_count': 20,
        'maximum_stable_negative_count': 19,
        'only_when_hold_action_and_negative_class_lost': True,
    }:
        raise ValueError('saturation-triggered tightening policy drift')

# The API environment has pydantic; the GPU environment deliberately does not.
if 'gpu-evaluate' not in sys.argv:
    from adm.preferences.gemini_vertex import GeminiVertexClient
    from adm.preferences.litellm_responses import LiteLLMResponsesClient

    class RecoveringGeminiClient(GeminiVertexClient):
        """Up to two logged retries, never a retry of valid negative judgments."""
        maximum_real_calls = 700

        def _completed_usage(self):
            prompt, output = 0, 0
            # Missing responses/usage are conservatively charged their full
            # reservations; they cannot wedge or silently undercount recovery.
            for path in self.attempt_root.glob('*/attempt_started.json'):
                response_path=path.parent/'provider_response.json'
                if response_path.exists():
                    envelope=read(response_path)
                    status=envelope.get('http_status')
                    if not isinstance(status,int): raise ValueError('invalid preserved HTTP status')
                    if not 200<=status<300: continue
                    try:
                        used_prompt,used_output=self._usage_counts(envelope.get('provider_response') or {})
                        prompt+=used_prompt; output+=used_output
                        continue
                    except ValueError:
                        pass
                projection = read(path).get('budget_projection') or {}
                prompt += int(projection.get('next_prompt_token_upper_bound', 100000))
                output += int(projection.get('next_maximum_output_tokens', 32768))
            return prompt, output

        def parse(self, request, response_model):
            original = request
            for retry in range(3):
                if retry:
                    context = json.loads(original.logical_context_json)
                    context['operational_retry'] = retry
                    request = replace(original, logical_context_json=json.dumps(context))
                folder = self.attempt_root / request.request_hash
                if not folder.exists() and len(list(self.attempt_root.glob('*/attempt_started.json'))) >= self.maximum_real_calls:
                    raise PermissionError('real API attempt ceiling reached')
                try:
                    response = super().parse(request, response_model)
                    # Logical artifact keys stay those of the original request;
                    # raw response and all retry hashes remain in the ledger.
                    if retry:
                        return replace(response, request_hash=original.request_hash,
                            raw_response={**response.raw_response, 'durable_retry_request_hash': request.request_hash})
                    return response
                except Exception as error:
                    message = str(error)
                    envelope = read(folder / 'provider_response.json') if (folder / 'provider_response.json').exists() else {}
                    status = envelope.get('http_status')
                    raw = envelope.get('provider_response') or {}
                    blocked = raw.get('promptFeedback', {}).get('blockReason') or any(
                        row.get('finishReason') in {'SAFETY', 'PROHIBITED_CONTENT'} for row in raw.get('candidates', [])) or any(
                        part.get('type') == 'refusal' for row in raw.get('output', [])
                        for part in row.get('content', []))
                    ambiguity = (folder / 'attempt_started.json').exists() and not envelope
                    transient = status in {408, 429, 500, 502, 503, 504} or ambiguity
                    malformed = (folder / 'parse_failure.json').exists() and not blocked
                    fatal = isinstance(error, PermissionError) or status in {400, 401, 403, 404} or any(
                        token in message for token in ('modelVersion', 'hash mismatch', 'previously parse-validated'))
                    if retry == 2 or fatal or not (transient or malformed): raise
                    write(self.attempt_root / f'recovery-{original.request_hash}-{retry}.json',
                          {'original_request_hash': original.request_hash, 'failed_request_hash': request.request_hash,
                           'next_retry': retry + 1, 'http_status': status, 'ambiguous': ambiguity,
                           'reason': type(error).__name__, 'scientific_payload_unchanged': True})
                    time.sleep(5 * (retry + 1))

    class RecoveringLiteLLMClient(RecoveringGeminiClient, LiteLLMResponsesClient):
        """Retain the registered retry policy with gateway billing reservations."""
        _completed_usage = LiteLLMResponsesClient._completed_usage


def endpoint_reserve_reached():
    deadline=float(os.environ.get('ADM_ALLOCATION_DEADLINE_EPOCH', 'inf'))
    return deadline-time.time() < 7200







def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    from adm.artifacts import immutable_write
    immutable_write(Path(path), json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
