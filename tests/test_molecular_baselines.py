"""Offline baseline contract and objective tests; synthetic tensors only."""

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from domains.molecules.gsk3b.artifacts import immutable_json, sha256_file
from domains.molecules.gsk3b.baselines import manifest as legacy
from domains.molecules.gsk3b.baselines import nft_loss, nft_manifest, nft_update
from domains.molecules.gsk3b.config import load_campaign_config
from domains.molecules.gsk3b.oracle import MODEL_ID, REWARD_ENDPOINT, HELD_OUT_EVALUATOR_IDS
from domains.molecules.gsk3b import reward as reward_module

ROOT = Path(__file__).resolve().parents[1]


class BaselineContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rollout = {
            'arm': 'nft', 'seed': 1, 'collection_checkpoint': '/mock/parent',
            'samples': [{'candidate_id': 'one', 'group_id': 'g0',
                         'content_sha256': 'native', 'training_state': {}}],
        }
        path = self.root / 'rollout.json'
        immutable_json(path, self.rollout)
        self.reward = {
            'reward_model_id': MODEL_ID, 'reward_endpoint': REWARD_ENDPOINT,
            'rollout': str(path), 'rollout_sha256': sha256_file(path),
            'failed_candidates_scored': True, 'usable_as_held_out_evaluator': False,
            'samples': [{'candidate_id': 'one', 'group_id': 'g0',
                         'raw_reward': 0.25, 'scored_by_model': True}],
        }

    def build(self):
        return nft_manifest.manifest(self.rollout, self.reward, 'scientific final goal')

    def test_manifest_retains_canonical_reward_and_held_out_identity(self):
        result = self.build()
        self.assertEqual(result['reward_model_id'], MODEL_ID)
        self.assertEqual(result['reward_endpoint'], REWARD_ENDPOINT)
        self.assertEqual(result['held_out_evaluator_ids'], list(HELD_OUT_EVALUATOR_IDS))
        self.assertFalse(result['scientific_qualification_passed'])

    def test_foreign_reward_cannot_be_relabelled_as_gsk3b(self):
        for field in ('reward_model_id', 'reward_endpoint'):
            with self.subTest(field=field):
                reward = {**self.reward, field: 'foreign-target'}
                with self.assertRaisesRegex(ValueError, 'registered GSK3B'):
                    nft_manifest.manifest(self.rollout, reward, 'goal')

    def test_reward_cannot_join_another_cohort_with_same_ids(self):
        altered = copy.deepcopy(self.rollout)
        altered['samples'][0]['content_sha256'] = 'different-native'
        with self.assertRaisesRegex(ValueError, 'rollout'):
            nft_manifest.manifest(altered, self.reward, 'goal')

    def test_changed_rollout_file_rejected(self):
        Path(self.reward['rollout']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'rollout'):
            self.build()

    def test_unbound_legacy_reward_rejected(self):
        self.reward.pop('rollout_sha256')
        with self.assertRaisesRegex(ValueError, 'rollout'):
            self.build()

    def test_duplicate_ids_do_not_collapse_silently(self):
        for name in ('rollout', 'reward'):
            with self.subTest(source=name):
                rollout, reward = copy.deepcopy(self.rollout), copy.deepcopy(self.reward)
                obj = rollout if name == 'rollout' else reward
                obj['samples'] *= 2
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    legacy.join_rewards(rollout, reward)

    def test_empty_or_incomplete_cohort_rejected(self):
        for rollout, reward in (({'samples': []}, {'samples': []}),
                                (self.rollout, {'samples': []})):
            with self.assertRaises(ValueError):
                legacy.join_rewards(rollout, reward)

    def test_group_change_rejected(self):
        self.reward['samples'][0]['group_id'] = 'different'
        with self.assertRaisesRegex(ValueError, 'different group'):
            self.build()

    def test_rewards_must_be_finite_probabilities(self):
        for value in (float('nan'), float('inf'), -0.1, 1.1, True, '0.3'):
            with self.subTest(value=value):
                self.reward['samples'][0]['raw_reward'] = value
                with self.assertRaisesRegex(ValueError, 'probability'):
                    self.build()

    def test_legacy_manifest_api_uses_same_contract(self):
        result = legacy.build_manifest(
            rollout=self.rollout, reward=self.reward, training_goal=REWARD_ENDPOINT,
            immutable_final_goal='scientific final goal',
            samples=legacy.join_rewards(self.rollout, self.reward))
        self.assertEqual(result, self.build())

    def test_adapter_and_campaign_use_same_reward(self):
        adapter = json.loads((ROOT / 'domains/molecules/gsk3b/configs/preference_training/flowmol_diffusion_nft_v1.json').read_text())
        config = load_campaign_config(ROOT / 'domains/molecules/gsk3b/baselines/nft.json')
        contract = nft_manifest.contract(config, '/mock/parent')
        self.assertEqual(contract, legacy.nft_contract_from_campaign(config, collection_checkpoint='/mock/parent'))
        for value in (adapter, contract):
            self.assertEqual(value['reward_model_id'], MODEL_ID)
            self.assertEqual(value['reward_endpoint'], REWARD_ENDPOINT)

    def test_reward_writer_emits_rollout_binding_and_keeps_failures(self):
        oracle = self.root / 'oracle'
        oracle.write_text('mock oracle')
        with patch.object(reward_module, 'load_oracle', return_value=object()):
            result = reward_module.score_rollout(Path(self.reward['rollout']), oracle, self.root / 'reward', 0.0)
        self.assertEqual(result['rollout_sha256'], self.reward['rollout_sha256'])
        self.assertEqual(result['sample_count'], 1)
        self.assertEqual(result['floored_count'], 1)
        nft_manifest.manifest(self.rollout, result, 'goal')


class NFTObjectiveTests(unittest.TestCase):
    def test_finite_optimality_and_tie_neutrality(self):
        rewards = torch.tensor([0.0, 0.5, 1.0, 0.2, 0.2])
        values = nft_loss.optimality_reward(rewards, ['g', 'g', 'g', 'h', 'h'])
        torch.testing.assert_close(values, torch.tensor([0.0, 0.5, 1.0, 0.5, 0.5]))

    def test_zero_reward_variance_is_neutral(self):
        values = nft_loss.optimality_reward(torch.ones(4), ['g'] * 4)
        torch.testing.assert_close(values, torch.full((4,), 0.5))

    def test_identical_policy_implicit_predictions_are_identical(self):
        values = torch.tensor([1.0, 2.0])
        positive, negative = nft_loss.implicit_policies(values, values, beta=1.0)
        torch.testing.assert_close(positive, values)
        torch.testing.assert_close(negative, values)

    def test_mutating_vector_field_cannot_change_reference_input(self):
        graph = SimpleNamespace(batch_size=1, marker=0)
        observations = []
        def forward(candidate, *args, **kwargs):
            observations.append(candidate.marker)
            candidate.marker += 1
            return {'x': torch.tensor([float(candidate.marker)])}
        policy = SimpleNamespace(vector_field=forward)
        utility = SimpleNamespace(get_batch_idxs=lambda graph: (None, None),
                                  get_upper_edge_mask=lambda graph: None)
        with patch.dict(sys.modules, {'flowmol.data_processing.utils': utility}), \
             patch.object(nft_update, 'copy_dgl_graph', side_effect=copy.deepcopy), \
             patch.object(nft_update, 'per_graph_flow_loss', side_effect=lambda *args, **kw: kw['vf_output']['x']):
            nft_update.flowmol_nft_step(policy, policy, graph, torch.ones(1),
                                       optimality=torch.tensor([0.5]), beta=1.0)
        self.assertEqual(observations, [0, 0])
        self.assertEqual(graph.marker, 0)


class NFTPreflightTests(unittest.TestCase):
    def setUp(self):
        self.contract = {'algorithm': 'diffusion_nft_v1', 'reward_model_id': MODEL_ID,
                         'collection_checkpoint': 'parent'}
        self.manifest = {'reward_model_id': MODEL_ID, 'collection_checkpoint': 'parent',
                         'fresh_rollout': True, 'training_state_bindings_verified': True,
                         'failed_candidates_scored': True,
                         'held_out_evaluator_ids': list(HELD_OUT_EVALUATOR_IDS),
                         'samples': [{'content_sha256': 'digest', 'group_id': 'g0', 'raw_reward': 0.3}]}

    def test_valid_baseline_passes(self):
        self.assertTrue(nft_loss.validate_nft_preflight(self.contract, self.manifest).ready)

    def test_reward_cannot_be_its_own_held_out_evaluator(self):
        for key, value in (('held_out_evaluator_id', MODEL_ID), ('held_out_evaluator_ids', [MODEL_ID])):
            with self.subTest(key=key):
                result = nft_loss.validate_nft_preflight(self.contract, {**self.manifest, key: value})
                self.assertIn('reward_model_is_the_held_out_evaluator', result.blockers)

    def test_missing_group_is_not_a_real_reward_group(self):
        self.manifest['samples'][0].pop('group_id')
        self.assertFalse(nft_loss.validate_nft_preflight(self.contract, self.manifest).ready)

    def test_nonfinite_reward_fails_preflight(self):
        self.manifest['samples'][0]['raw_reward'] = float('nan')
        self.assertFalse(nft_loss.validate_nft_preflight(self.contract, self.manifest).ready)


if __name__ == '__main__':
    unittest.main()
