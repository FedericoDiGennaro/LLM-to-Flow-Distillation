"""Offline integration contracts for the added DRD2 and kinase workflows."""
import contextlib
import copy
import datetime as dt
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from adm import cli

ROOT = Path(__file__).resolve().parents[1]
TASKS = ('drd2', 'kinase')


def module(task, name):
    return importlib.import_module(f'domains.molecules.{task}.{name}')


def judgment(goal, *, abstain=0.0):
    return {'goal_id': goal, 'counts': {'positive': 40, 'negative': 60, 'abstain': 0, 'unstable': 0},
            'q_positive_over_all_attempts': 0.4, 'exact_agreement': 1.0,
            'abstain_fraction_by_pass': [abstain, abstain]}


def controller(task, *, gate_passed=True, final_abstain=0.0):
    run, gates = module(task, 'run'), module(task, 'gates')
    config_path = ROOT / f'domains/molecules/{task}/configs/dpo.json'
    config = json.loads(config_path.read_text())
    config['curriculum_loop']['maximum_alignment_rounds'] = 1
    config['geometric_control']['maximum_total_updates'] = 3
    config['final_f_continuation']['enabled'] = True
    config['final_f_continuation']['maximum_updates'] = 2
    config['heuristic_control']['working_F_stop_enabled'] = False
    cohort = {'generation': Path('/mock/generation'), 'rows': [], 'pool': Path('/mock/pool')}
    initial = judgment('F-p0')
    initial['counts'].update(positive=0, negative=100)
    initial['q_positive_over_all_attempts'] = 0.0
    responses = [initial, judgment('G01-entry'), judgment('G01-child'),
                 judgment('F-child'), judgment('F-working'), judgment('G-working'),
                 judgment('F-final-child'), judgment('F-final-working', abstain=final_abstain),
                 judgment('F-final-child2'), judgment('F-final-working2')]
    saved = {}
    argv = ['run', '--config', str(config_path), '--output-root', '/mock/campaign',
            '--checkpoint', '/mock/parent', '--n-atoms-hist', '/mock/hist',
            '--marginal-dists', '/mock/marginal', '--flowmol-python', '/mock/python',
            '--authorization', '/mock/auth', '--active-campaign', '/mock/active']
    with contextlib.ExitStack() as stack:
        for name, value in {
            'load_campaign_config': Mock(return_value=config),
            'verify_authorization': Mock(),
            'bind_configuration': Mock(return_value=(config, Path('/mock/frozen.json'))),
            'RoleExecutor': Mock(return_value=SimpleNamespace(calls=0)),
            'generate_cohort': Mock(return_value=cohort),
            'judge_cohort': Mock(side_effect=responses),
            'existing_goal_proposal': Mock(return_value=None),
            'propose_goal': Mock(return_value=(SimpleNamespace(proposed_goal='observable precursor'), None, True)),
            'immutable_json': Mock(side_effect=lambda p,v: saved.update({str(p): v})),
            'run_dpo_update': Mock(return_value=(Path('/mock/child'), Path('/mock/training.json'))),
            '_run': Mock(side_effect=AssertionError('No subprocess in offline controller test')),
        }.items():
            stack.enter_context(patch.object(run, name, value))
        receipt = {'all_engineering_gates_passed': gate_passed,
                   'checks': {name: gate_passed for name in gates.NONCOLLAPSE_CHECKS},
                   **{name+'_sha256': 'digest' for name in ('parent_checkpoint', 'child_checkpoint',
                         'parent_samples', 'frozen_p0_samples', 'child_samples')}}
        stack.enter_context(patch.object(gates, 'read_json', return_value=receipt))
        stack.enter_context(patch.object(gates, 'sha256_file', return_value='digest'))
        stack.enter_context(patch.object(Path, 'mkdir'))
        stack.enter_context(patch.object(Path, 'is_file', return_value=True))
        stack.enter_context(patch('sys.argv', argv))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        run.main()
        return saved['/mock/campaign/campaign_result.json'], run.run_dpo_update.call_count


class AddedControllerTests(unittest.TestCase):
    def test_saved_failed_gate_stops_before_judging_child(self):
        for task in TASKS:
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'noncollapse gate'):
                controller(task, gate_passed=False)

    def test_final_continuation_rechecks_abstentions(self):
        for task in TASKS:
            with self.subTest(task=task):
                result, updates = controller(task, final_abstain=0.6)
                self.assertEqual(updates, 2)
                self.assertEqual(result['final_f_update_count'], 1)
                result, updates = controller(task)
                self.assertEqual(updates, 3)
                self.assertEqual(result['final_f_update_count'], 2)

    def test_receipts_bind_samples_and_boolean_checks(self):
        for task in TASKS:
            gates = module(task, 'gates')
            receipt = {'all_engineering_gates_passed': True,
                       'checks': {name: True for name in gates.NONCOLLAPSE_CHECKS},
                       'child_samples_sha256': 'old'}
            with self.subTest(task=task):
                self.assertTrue(module(task, 'validate')._noncollapse_valid(receipt))
                bad = copy.deepcopy(receipt)
                bad['checks']['prior_policy_satisfied'] = 'false'
                self.assertFalse(module(task, 'validate')._noncollapse_valid(bad))
                with patch.object(gates, 'read_json', return_value=receipt), \
                     patch.object(gates, 'sha256_file', return_value='new'), \
                     self.assertRaisesRegex(ValueError, 'binding differs'):
                    gates.require_noncollapse(Path('/mock/gate'), inputs={'child_samples': Path('/mock/sample')})

    def test_task_specific_advancement_rules(self):
        drd2 = module('drd2', 'resume')
        kinase = module('kinase', 'resume')
        self.assertTrue(drd2.goal_passed({'minimum_enriched_positive_fraction': .3, 'minimum_enrichment': .1},
                                         {'child_q': .6, 'child_minus_adoption': .6-.5}))
        control = {'minimum_enriched_positive_fraction': .2, 'enrichment_rule': 'strict_absolute_fraction'}
        self.assertFalse(kinase.goal_passed(control, {'child_q': .2}))
        self.assertTrue(kinase.goal_passed(control, {'child_q': .201}))

    def test_resume_rejects_changed_provider_revision(self):
        for task in TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root/'candidate_judge/api_attempts/attempt-1/provider_response.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'http_status': 200, 'provider_response': {'modelVersion': 'old'}}))
                config = json.loads((ROOT/f'domains/molecules/{task}/configs/dpo.json').read_text())
                with patch.dict('os.environ', {'ADM_LLM_PROVIDER': 'gemini'}):
                    executor = module(task, 'run').RoleExecutor(role='candidate_judge', config=config, root=root)
                with self.assertRaisesRegex(ValueError, 'modelVersion changed'):
                    executor.client.observe_model_version('new')

    def test_authorization_rejects_foreign_scope_and_changed_code(self):
        for task in TASKS:
            run = module(task, 'run')
            config = {'experiment_id': 'test'}
            auth = {'status': 'approved', 'experiment_id': 'test', 'output_root': '/mock/campaign',
                    'config_file_sha256': 'profile', 'authorization_scope': 'full_ltfd',
                    'code_commit': 'commit', 'resolved_config_sha256': hashlib.sha256(run.canonical_json(config).encode()).hexdigest(),
                    'deadline_utc': (dt.datetime.now(dt.timezone.utc)+dt.timedelta(days=1)).isoformat()}
            for scope, commit in [('full_ltfd', 'commit'), ('full_ltfd', 'different'), (f'{task}_only_dpo_baseline', 'commit')]:
                auth['authorization_scope'] = scope
                campaign = {**auth, 'status': 'active'}
                with self.subTest(task=task, scope=scope, commit=commit), \
                     patch.object(run, 'read_json', side_effect=[auth,campaign]), \
                     patch.object(run, 'sha256_file', return_value='profile'), \
                     patch.object(run.subprocess, 'check_output', side_effect=[commit,'']):
                    if scope=='full_ltfd' and commit=='commit':
                        run.verify_authorization(Path('/mock/config'), config, Path('/mock/campaign'), Path('/mock/auth'), Path('/mock/active'))
                    else:
                        with self.assertRaises(PermissionError):
                            run.verify_authorization(Path('/mock/config'), config, Path('/mock/campaign'), Path('/mock/auth'), Path('/mock/active'))


class AddedBaselineTests(unittest.TestCase):
    def test_reward_binding_and_invalid_reward_rows(self):
        for task in TASKS:
            oracle, manifest = module(task, 'oracle'), module(task, 'baselines.nft_manifest')
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'rollout.json'
                rollout = {'arm': 'nft', 'seed': 1, 'collection_checkpoint': '/mock/parent',
                           'samples': [{'candidate_id':'one','group_id':'g0','content_sha256':'native','training_state':{}}]}
                path.write_text(json.dumps(rollout))
                reward = {'reward_model_id':oracle.MODEL_ID,'reward_endpoint':oracle.REWARD_ENDPOINT,
                          'rollout':str(path),'rollout_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'failed_candidates_scored':True,'usable_as_held_out_evaluator':False,
                          'samples':[{'candidate_id':'one','group_id':'g0','raw_reward':.25,'scored_by_model':True}]}
                self.assertEqual(manifest.manifest(rollout,reward,'goal')['reward_model_id'],oracle.MODEL_ID)
                variants = [{**reward,'reward_model_id':'foreign'}, {**reward,'rollout_sha256':'different'},
                            {**reward,'samples':reward['samples']*2}]
                for value in [float('nan'),float('inf'),-.1,1.1]:
                    variants.append({**reward,'samples':[{**reward['samples'][0],'raw_reward':value}]})
                for bad in variants:
                    with self.assertRaises(ValueError):manifest.manifest(rollout,bad,'goal')

    def test_vector_field_inputs_do_not_leak_to_reference(self):
        for task in TASKS:
            update = module(task, 'baselines.nft_update')
            graph = SimpleNamespace(batch_size=1,marker=0)
            seen=[]
            def forward(candidate,*args,**kwargs):
                seen.append(candidate.marker); candidate.marker+=1
                return {'x':torch.tensor([float(candidate.marker)])}
            policy=SimpleNamespace(vector_field=forward)
            utility=SimpleNamespace(get_batch_idxs=lambda g:(None,None),get_upper_edge_mask=lambda g:None)
            with self.subTest(task=task), patch.dict(sys.modules,{'flowmol.data_processing.utils':utility}), \
                 patch.object(update,'copy_dgl_graph',side_effect=copy.deepcopy), \
                 patch.object(update,'per_graph_flow_loss',side_effect=lambda *args,**kw:kw['vf_output']['x']):
                update.flowmol_nft_step(policy,policy,graph,torch.ones(1),optimality=torch.tensor([.5]),beta=1.)
                self.assertEqual(seen,[0,0]);self.assertEqual(graph.marker,0)

    def test_reward_writer_binds_rollout(self):
        for task in TASKS:
            reward = module(task, 'reward')
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'rollout.json';path.write_text(json.dumps({'samples':[{'candidate_id':'one','group_id':'g0'}]}))
                with patch.object(reward,'load_oracle',return_value=None), patch.object(reward,'shared',return_value={}):
                    result=reward.score_rollout(path,Path('/mock/oracle'),Path(directory)/'reward',0.)
                self.assertEqual(result['rollout_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())


class TaskDefinitionTests(unittest.TestCase):
    def test_config_and_teacher_goals_agree(self):
        for task in TASKS:
            for profile in ('dpo', 'smoke'):
                with self.subTest(task=task, profile=profile):
                    config = module(task, 'config').load_campaign_config(ROOT/f'domains/molecules/{task}/configs/{profile}.json')
                    self.assertEqual(config['final_goal'], module(task, 'cards').FINAL_GOAL)
                    self.assertFalse(config['fresh_sampling']['same_native_graphs_for_parent_and_child'])

    def test_drd2_reward_keeps_logp_gate_and_median_split(self):
        oracle = module('drd2', 'oracle')
        self.assertEqual(oracle.combined_reward(.8, 2.), .8)
        self.assertEqual(oracle.combined_reward(.8, 4.), 0.)
        labels = oracle.induce_preference_labels(['a','b','c','d'], {'a':.1,'b':.2,'c':.7,'d':.9})
        self.assertEqual(labels, {'a':'negative','b':'negative','c':'positive','d':'positive'})

    def test_kinase_graph_reward_preserves_link_relaxation(self):
        oracle = module('kinase', 'oracle')
        for smiles, expected in [
            ('Nc1ncccc1C1CCNCC1',1),  # Carbon linkage.
            ('Nc1nccc(N2CCNCC2)c1',1),  # Heteroatom linkage, with a free ring NH.
            ('Nc1ncccc1C1CCN(C)CC1',0),  # Tertiary amine is not a free secondary amine.
            ('c1ncccc1C1CCNCC1',0),  # Missing primary amino group.
            ('Nc1ncccc1.C1CCNCC1',0),  # Disconnected fragments.
        ]:
            with self.subTest(smiles=smiles):
                self.assertEqual(oracle.score_smiles(smiles),expected)

    def test_added_tasks_use_selected_litellm_provider(self):
        from adm.preferences.litellm_responses import LiteLLMResponsesClient
        for task in TASKS:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory, \
                 patch.dict('os.environ', {'ADM_LLM_PROVIDER':'litellm', 'LITELLM_API_BASE':'https://gateway.example', 'LITELLM_API_KEY':'synthetic-key'}, clear=True):
                config=json.loads((ROOT/f'domains/molecules/{task}/configs/dpo.json').read_text())
                executor=module(task,'run').RoleExecutor(role='candidate_judge',config=config,root=Path(directory))
                self.assertIsInstance(executor.client,LiteLLMResponsesClient)


class LauncherTests(unittest.TestCase):
    def test_every_molecular_task_uses_its_own_profile(self):
        for task in ('gsk3b',*TASKS):
            for smoke in (False,True):
                with self.subTest(task=task,smoke=smoke), tempfile.TemporaryDirectory() as directory:
                    root=Path(directory);receipt=root/'assets/flowmol3/setup_receipt.json';receipt.parent.mkdir(parents=True)
                    receipt.write_text(json.dumps({'checkpoint':'/mock/model','n_atoms_hist_file':'/mock/hist',
                     'marginal_dists_file':'/mock/marginal','runtime_python':'/mock/python','source':'/mock/flowmol'}))
                    args=cli.build_parser().parse_args(['run',task,'--output-dir','/mock/output','--runtime-home',str(root)]+(['--smoke'] if smoke else []))
                    preparation=cli.preparation(args,ROOT,Path(sys.executable));steps,_=cli.molecular_steps(args,ROOT,Path(sys.executable))
                    expected=str(ROOT/f'domains/molecules/{task}/configs'/('smoke.json' if smoke else 'dpo.json'))
                    self.assertEqual(preparation[preparation.index('--config')+1],expected)
                    self.assertEqual(steps[0][1][steps[0][1].index('--config')+1],expected)
                    self.assertIn(f'domains.molecules.{task}.run',steps[0][1])


if __name__=='__main__':unittest.main()
