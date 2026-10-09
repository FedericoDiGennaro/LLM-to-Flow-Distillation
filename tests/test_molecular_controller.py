"""Offline controller regression tests; no provider, GPU or training work."""
import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import datetime as dt
import hashlib

from domains.molecules.gsk3b import gates
from unittest.mock import Mock, patch

from domains.molecules.gsk3b import run

ROOT = Path(__file__).resolve().parents[1]


def labels(goal_id, positives, negatives=60, agreement=1.0):
    return {
        'goal_id': goal_id,
        'counts': {'positive': positives, 'negative': negatives,
                   'abstain': 0, 'unstable': 100 - positives - negatives},
        'q_positive_over_all_attempts': positives / 100,
        'exact_agreement': agreement,
        'abstain_fraction_by_pass': [0.0, 0.0],
    }


def mocked_controller(child_labels, rounds=1, entry_positives=40, *, gate_passed=True,
                      final_labels=None, final_update_labels=None):
    config = json.loads((ROOT / 'domains/molecules/gsk3b/configs/dpo.json').read_text())
    config['curriculum_loop']['maximum_alignment_rounds'] = rounds
    config['geometric_control']['maximum_total_updates'] = rounds
    config['final_f_continuation']['enabled'] = final_update_labels is not None
    config['final_f_continuation']['maximum_updates'] = 2
    config['heuristic_control']['working_F_stop_enabled'] = False
    cohort = {'generation': Path('/mock/generation'), 'rows': [], 'pool': Path('/mock/pool')}
    initial = labels('F-p0', 0, 100)
    entry = labels('G01-entry', entry_positives, 100 - entry_positives)
    responses = [initial, entry]
    for index in range(rounds):
        responses.extend([child_labels, final_labels or labels(f'F-p{index+1}', 0, 100)])
    if final_update_labels is not None:
        responses.extend([final_update_labels, final_update_labels])
    saved = {}
    argv = ['run', '--config', str(ROOT / 'domains/molecules/gsk3b/configs/dpo.json'),
            '--output-root', '/mock/campaign', '--checkpoint', '/mock/parent.ckpt',
            '--n-atoms-hist', '/mock/hist.pt', '--marginal-dists', '/mock/marginal.pt',
            '--flowmol-python', '/mock/python', '--authorization', '/mock/authorization',
            '--active-campaign', '/mock/active']
    def capture(path, value):
        saved[str(path)] = value
    with contextlib.ExitStack() as stack:
        for target, value in [
            ('load_campaign_config', Mock(return_value=config)),
            ('verify_authorization', Mock()),
            ('bind_configuration', Mock(return_value=(config, Path('/mock/frozen.json')))),
            ('RoleExecutor', Mock(return_value=SimpleNamespace(calls=0))),
            ('generate_cohort', Mock(return_value=cohort)),
            ('judge_cohort', Mock(side_effect=responses)),
            ('existing_goal_proposal', Mock(return_value=None)),
            ('propose_goal', Mock(return_value=(SimpleNamespace(proposed_goal='observable precursor'), None, True))),
            ('immutable_json', Mock(side_effect=capture)),
            ('run_dpo_update', Mock(return_value=(Path('/mock/child.ckpt'), Path('/mock/training.json')))),
            ('_run', Mock(side_effect=AssertionError('No subprocess may execute in a review'))),

        ]:
            stack.enter_context(patch.object(run, target, value))
        receipt = {'all_engineering_gates_passed': gate_passed,
                   'checks': {name: gate_passed for name in gates.NONCOLLAPSE_CHECKS},
                   **{name + '_sha256': 'digest' for name in (
                       'parent_checkpoint', 'child_checkpoint', 'parent_samples',
                       'frozen_p0_samples', 'child_samples')}}
        stack.enter_context(patch.object(gates, 'read_json', return_value=receipt))
        stack.enter_context(patch.object(gates, 'sha256_file', return_value='digest'))
        stack.enter_context(patch('sys.argv', argv))
        stack.enter_context(patch.object(Path, 'mkdir'))
        # A prior validator invocation wrote a failing noncollapse receipt.
        stack.enter_context(patch.object(Path, 'is_file', return_value=True))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        run.main()
        return config, saved, run.run_dpo_update.call_args_list, run.judge_cohort.call_count


class ControllerRegressionTests(unittest.TestCase):
    def test_failed_saved_gate_blocks_child_judging(self):
        with self.assertRaisesRegex(ValueError, 'noncollapse gate'):
            mocked_controller(labels('G01-p1', 40), gate_passed=False)

    def test_passing_saved_gate_allows_judging(self):
        _, saved, updates, calls = mocked_controller(labels('G01-p1', 40))
        self.assertEqual(len(updates), 1)
        self.assertEqual(calls, 4)
        self.assertEqual(saved['/mock/campaign/campaign_result.json']['completed_round_count'], 1)

    def test_held_goal_rechecks_support_and_agreement(self):
        for weak in (labels('G01-p1', 1, 60, 1.0), labels('G01-p1', 30, 60, 0.4)):
            with self.subTest(weak=weak):
                config, saved, updates, _ = mocked_controller(weak, rounds=2)
                self.assertFalse(run.learnable(weak, config))
                self.assertEqual(len(updates), 1)
                self.assertEqual(saved['/mock/campaign/campaign_result.json']['curriculum_stop_reason'],
                                 'held_goal_no_longer_learnable')

    def test_held_learnable_goal_continues(self):
        _, _, updates, _ = mocked_controller(labels('G01-p1', 40), rounds=2)
        self.assertEqual(len(updates), 2)

    def test_inclusive_enrichment_boundary(self):
        self.assertLess(60 / 100 - 50 / 100, 0.1)
        self.assertTrue(run.clears_unlock(60 / 100 - 50 / 100, 0.1, {}, 1))
        _, saved, _, _ = mocked_controller(labels('G01-p1', 60, 40), entry_positives=50)
        self.assertTrue(saved['/mock/campaign/rounds/round-01/round_result.json']['enriched'])

    def test_final_goal_updates_recheck_agreement(self):
        _, saved, updates, _ = mocked_controller(
            labels('G01-p1', 40), final_labels=labels('F-p1', 40),
            final_update_labels=labels('F-final-p1', 40, 60, 0.4))
        self.assertEqual(len(updates), 2)  # One curriculum and one final-F update.
        self.assertEqual(saved['/mock/campaign/campaign_result.json']['final_f_update_count'], 1)

    def test_final_goal_updates_recheck_abstention(self):
        weak = labels('F-final-p1', 40)
        weak['abstain_fraction_by_pass'] = [0.6, 0.6]
        _, saved, updates, _ = mocked_controller(
            labels('G01-p1', 40), final_labels=labels('F-p1', 40), final_update_labels=weak)
        self.assertEqual(len(updates), 2)
        self.assertEqual(saved['/mock/campaign/campaign_result.json']['final_f_update_count'], 1)


class GateRegressionTests(unittest.TestCase):
    def test_inclusive_threshold_does_not_relax_meaningfully(self):
        self.assertTrue(gates.at_least(0.6 - 0.5, 0.1))
        self.assertFalse(gates.at_least(0.099, 0.1))
        for bad in (float('nan'), float('inf'), -float('inf')):
            self.assertFalse(gates.at_least(bad, 0.1))

    def test_explicit_historical_amendment_preserves_prefix(self):
        config = {'resume_revision': {'effective_from_round': 3}}
        self.assertFalse(run.clears_unlock(0.6 - 0.5, 0.1, config, 2))
        self.assertTrue(run.clears_unlock(0.6 - 0.5, 0.1, config, 3))

    def test_receipt_requires_each_boolean_gate(self):
        receipt = {'all_engineering_gates_passed': True,
                   'checks': {name: True for name in gates.NONCOLLAPSE_CHECKS}}
        self.assertTrue(gates.noncollapse_valid(receipt))
        for value in (False, 1, 'true', None):
            bad = copy.deepcopy(receipt)
            bad['checks']['checkpoint_changed'] = value
            self.assertFalse(gates.noncollapse_valid(bad))
        for checks in ({}, None, []):
            self.assertFalse(gates.noncollapse_valid({**receipt, 'checks': checks}))

    def test_receipt_must_bind_actual_samples(self):
        receipt = {'all_engineering_gates_passed': True,
                   'checks': {name: True for name in gates.NONCOLLAPSE_CHECKS},
                   'child_samples_sha256': 'old'}
        with patch.object(gates, 'read_json', return_value=receipt), \
             patch.object(gates, 'sha256_file', return_value='new'):
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                gates.require_noncollapse(Path('/mock/gate'), inputs={'child_samples': Path('/mock/samples')})

    def test_missing_receipt_is_not_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                gates.require_noncollapse(Path(directory) / 'missing.json', inputs={})


class ModelVersionRegressionTests(unittest.TestCase):
    def make_executor(self, root):
        config = json.loads((ROOT / 'domains/molecules/gsk3b/configs/dpo.json').read_text())
        return run.RoleExecutor(role='candidate_judge', config=config, root=root)

    def test_resume_restores_provider_revision_before_new_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'candidate_judge/api_attempts/attempt-1/provider_response.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'http_status': 200, 'provider_response': {'modelVersion': 'old'}}))
            executor = self.make_executor(root)
            with self.assertRaisesRegex(ValueError, 'modelVersion changed'):
                executor.client.observe_model_version('new')

    def test_resume_rejects_mixed_successful_revisions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, version in enumerate(('old', 'new')):
                path = root / f'candidate_judge/api_attempts/attempt-{index}/provider_response.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'http_status': 200, 'provider_response': {'modelVersion': version}}))
            with self.assertRaisesRegex(ValueError, 'modelVersion changed'):
                self.make_executor(root)

    def test_failed_http_response_does_not_bind_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'candidate_judge/api_attempts/attempt-1/provider_response.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'http_status': 429, 'provider_response': {}}))
            self.assertIsNone(self.make_executor(root).client._observed_model_version)

    def test_cached_response_restores_model_binding(self):
        from pydantic import BaseModel
        from domains.molecules.gsk3b.transport import GeminiVertexClient
        class Answer(BaseModel):
            value: int
        executor = run.RoleExecutor.__new__(run.RoleExecutor)
        executor.maximum_operational_retries = 0
        executor.maximum_calls = 5
        executor.calls = 0
        executor.store = Mock()
        executor.store.load_response.return_value = SimpleNamespace(
            parsed={'value': 1}, usage={},
            raw_response={'model_version': 'old', 'provider_response': {'modelVersion': 'old'}})
        executor.client = GeminiVertexClient(api_key_provider=lambda: 'unused')
        request = SimpleNamespace(request_hash='saved-request')
        executor.execute(request, Answer)
        self.assertEqual(executor.calls, 0)
        with self.assertRaisesRegex(ValueError, 'modelVersion changed'):
            executor.client.observe_model_version('new')


class AuthorizationRegressionTests(unittest.TestCase):
    def setUp(self):
        self.config = {'experiment_id': 'test'}
        self.authorization = {
            'status': 'approved', 'experiment_id': 'test',
            'output_root': '/mock/campaign', 'config_file_sha256': 'profile-digest',
            'authorization_scope': 'full_ltfd', 'code_commit': 'frozen-commit',
            'resolved_config_sha256': hashlib.sha256(run.canonical_json(self.config).encode()).hexdigest(),
            'deadline_utc': (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).isoformat(),
        }
        self.campaign = {**self.authorization, 'status': 'active'}

    def verify(self, commit='frozen-commit', dirty=''):
        with patch.object(run, 'read_json', side_effect=[self.authorization, self.campaign]), \
             patch.object(run, 'sha256_file', return_value='profile-digest'), \
             patch.object(run.subprocess, 'check_output', side_effect=[commit, dirty]):
            run.verify_authorization(Path('/mock/config'), self.config, Path('/mock/campaign'),
                                     Path('/mock/authorization'), Path('/mock/active'))

    def test_matching_clean_commit_passes(self):
        self.verify()

    def test_changed_code_or_dirty_tree_rejected(self):
        for commit, dirty in (('different-commit', ''), ('frozen-commit', ' M run.py')):
            with self.subTest(commit=commit, dirty=dirty):
                with self.assertRaisesRegex(PermissionError, 'authorized clean code'):
                    self.verify(commit, dirty)

    def test_foreign_campaign_receipt_rejected(self):
        self.campaign['experiment_id'] = 'unrelated'
        with self.assertRaisesRegex(PermissionError, 'campaign/authorization'):
            self.verify()

    def test_baseline_authorization_cannot_start_llm_curriculum(self):
        self.authorization['authorization_scope'] = 'gsk3b_only_dpo_baseline'
        self.campaign['authorization_scope'] = 'gsk3b_only_dpo_baseline'
        with self.assertRaisesRegex(PermissionError, 'scope'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
