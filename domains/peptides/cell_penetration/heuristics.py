"""Prospectively fixed EXP110 host rules; no predictor or provider access."""
from __future__ import annotations

import math

ARMS = ('gate60', 'gate50', 'gate40', 'gate50_patience', 'gate50_F30')


def policy_for_arm(arm):
    if arm not in ARMS:
        raise ValueError('unknown EXP110 arm')
    return {
        'monitor_working_F_every_update': True,
        'enrichment_floor': .6 if arm == 'gate60' else .4 if arm == 'gate40' else .5,
        'minimum_enrichment': .1,
        'maximum_updates_per_goal': 8 if arm == 'gate50_patience' else 5,
        'patience_enabled': arm == 'gate50_patience',
        'patience_updates': 3,
        'meaningful_progress_delta': .05,
        'working_F_stop_enabled': arm == 'gate50_F30',
        'working_F_floor': .30,
        'working_F_gain_over_p0': .10,
        'working_F_consecutive_cohorts': 2,
    }


def at_least(value, threshold):
    return value > threshold or math.isclose(value, threshold, rel_tol=0, abs_tol=1e-12)


def _check_rate(q):
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError('invalid stable-positive fraction')


def advance_patience(previous, *, goal_id, distribution_id, entry_q, q, policy):
    """Tiny improvements accumulate toward an anchor; cache replay is idempotent."""
    _check_rate(entry_q); _check_rate(q)
    if previous is None or previous['goal_id'] != goal_id:
        previous = {'goal_id': goal_id, 'entry_q': entry_q, 'anchor_q': entry_q,
                    'raw_best_q': entry_q, 'stagnant_updates': 0,
                    'last_distribution_id': None}
    if previous['entry_q'] != entry_q:
        raise ValueError('goal-entry judgment changed')
    if previous['last_distribution_id'] == distribution_id:
        if previous['current_q'] != q:
            raise ValueError('same cohort has changed surrogate judgment')
        return dict(previous)
    if previous['last_distribution_id'] is not None and (
        int(distribution_id[1:]) != int(previous['last_distribution_id'][1:]) + 1
    ):
        raise ValueError('patience updates must be consecutive generator cohorts')
    progressed = at_least(q - previous['anchor_q'], policy['meaningful_progress_delta'])
    return {**previous, 'last_distribution_id': distribution_id, 'current_q': q,
            'anchor_q': q if progressed else previous['anchor_q'],
            'raw_best_q': max(q, previous['raw_best_q']),
            'stagnant_updates': 0 if progressed else previous['stagnant_updates'] + 1}


def advance_final_monitor(previous, *, distribution_id, p0_q, q, policy):
    """Two qualifying consecutive post-update cohorts; each already judged twice."""
    _check_rate(p0_q); _check_rate(q)
    if previous is None:
        if distribution_id != 'p0' or q != p0_q:
            raise ValueError('F monitoring must initialize at p0')
        return {'p0_q': p0_q, 'current_q': q, 'last_distribution_id': 'p0',
                'qualifying_streak': 0, 'target_confirmed': False}
    if previous['p0_q'] != p0_q:
        raise ValueError('fixed p0 F judgment changed')
    if previous['last_distribution_id'] == distribution_id:
        if previous['current_q'] != q:
            raise ValueError('same cohort has changed F judgment')
        return dict(previous)
    if int(distribution_id[1:]) != int(previous['last_distribution_id'][1:]) + 1:
        raise ValueError('F monitoring must cover consecutive generator cohorts')
    qualifies = (at_least(q, policy['working_F_floor']) and
                 at_least(q - p0_q, policy['working_F_gain_over_p0']))
    streak = previous['qualifying_streak'] + 1 if qualifies else 0
    return {**previous, 'current_q': q, 'last_distribution_id': distribution_id,
            'qualifying_streak': streak,
            'target_confirmed': streak >= policy['working_F_consecutive_cohorts']}
