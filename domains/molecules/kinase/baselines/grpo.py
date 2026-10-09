#!/usr/bin/env python3
"""Collect stochastic FlowMol trajectories and run an on-policy FlowGRPO update."""
from __future__ import annotations
import argparse
from pathlib import Path
from domains.molecules.kinase.artifacts import immutable_json, read_json
from domains.molecules.kinase.config import load_frozen_or_campaign_config
from domains.molecules.kinase.oracle import MODEL_ID, REWARD_ENDPOINT
from domains.molecules.kinase.run import _run, generate_cohort, verify_authorization
from domains.molecules.kinase.baselines.dpo import score_pool, mean_attempt_reward

def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('config','output-root','checkpoint','n-atoms-hist','marginal-dists',
                 'flowmol-python','authorization','active-campaign','tdc-python','oracle'):
        result.add_argument('--'+name, type=Path, required=True)
    return result

def initialize(args):
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'frozen_config.json'
    config = load_frozen_or_campaign_config(args.config.resolve(), frozen)
    if config['preference_source'] != MODEL_ID:
        raise ValueError('training reward does not match this task')
    verify_authorization(args.config.resolve(), config, output,
                         args.authorization.resolve(), args.active_campaign.resolve(),
                         expected_scopes=("kinase_only_distilled_guidance_baseline",)
                         if config.get('algorithm') == 'time_conditioned_soft_graph_reward_guidance_v1'
                         else ("kinase_only_flow_grpo_baseline",))
    immutable_json(frozen, config)
    return config, output, frozen

def collect_and_score(args, config, output):
    spec = config['flowgrpo']
    rollout = output / 'rollout'
    if not (rollout / 'rollout.json').exists():
        command = [str(args.flowmol_python.absolute()), '-m', 'domains.molecules.kinase.sample',
            '--rollout', '--checkpoint', str(args.checkpoint.resolve()),
            '--n-atoms-hist', str(args.n_atoms_hist.resolve()),
            '--marginal-dists', str(args.marginal_dists.resolve()), '--output-dir', str(rollout)]
        for flag, key in (('count','rollout_count'),('batch-size','rollout_batch_size'),
                          ('group-size','group_size'),('transition-count','transition_count'),
                          ('noise-level','noise_level'),('categorical-stochasticity','categorical_stochasticity'),
                          ('seed','rollout_seed')):
            command += ['--'+flag, str(spec[key])]
        _run(command, timeout=28800)
    reward_root = output / 'training-reward'
    reward = reward_root / 'kinase_reward.json'
    if not reward.exists():
        _run([str(args.tdc_python.absolute()), '-m', 'domains.molecules.kinase.reward',
              '--rollout', str(rollout / 'rollout.json'), '--oracle', str(args.oracle.resolve()),
              '--output-dir', str(reward_root)], timeout=7200)
    data = read_json(reward)
    if data['reward_model_id'] != MODEL_ID or data.get('failed_candidates_scored') is not True:
        raise ValueError('rollout reward contract failed')
    if data['sample_count'] != int(spec['rollout_count']):
        raise ValueError('rollout reward omitted attempts')
    return rollout, reward, data

def no_signal(output, args, method, data, status="no_update_constant_reward"):
    immutable_json(output / 'campaign_result.json', {
        'method': method, 'status': status,
        'training_reward': REWARD_ENDPOINT, 'sample_count': data['sample_count'],
        'final_model': str(args.checkpoint.resolve()), 'provider_calls': 0,
        'generator_updates': 0,
    })

def evaluate(args, config, frozen, output, checkpoint):
    spec = config['fresh_sampling']
    cohort = generate_cohort(checkpoint=checkpoint, root=output / 'development',
        count=int(spec['num_attempts_per_arm']), batch_size=int(spec['batch_size']),
        seed=int(spec['round_1_seeds'][0]), n_timesteps=int(spec['flow_timesteps']),
        role='development', config_path=frozen, python=args.flowmol_python.absolute(),
        n_atoms_hist=args.n_atoms_hist.resolve(), marginal_dists=args.marginal_dists.resolve())
    reward = score_pool(tdc_python=args.tdc_python, pool=cohort['pool'] / 'candidate_pool.jsonl',
        oracle=args.oracle, output=output / 'evaluation', config=config, induce_preferences=False)
    return mean_attempt_reward(reward)

def main():
    args = parser().parse_args()
    config, output, frozen = initialize(args)
    if config['algorithm'] != 'flow_grpo_ode_to_sde_v1':
        raise ValueError('expected an ODE-to-SDE FlowGRPO profile')
    rollout, reward, data = collect_and_score(args, config, output)
    groups = {}
    for row in data['samples']:
        groups.setdefault(str(row['group_id']), set()).add(float(row['raw_reward']))
    if not any(len(values) > 1 for values in groups.values()):
        no_signal(output, args, 'flow_grpo_ode_to_sde_v1', data)
        return
    update = output / 'update'
    if not (update / 'training_result.json').exists():
        _run([str(args.flowmol_python.absolute()), '-m', 'domains.molecules.kinase.baselines.grpo_update',
            '--config', str(frozen), '--trajectory', str(rollout / 'trajectories.pt'),
            '--reward', str(reward), '--parent-checkpoint', str(args.checkpoint.resolve()),
            '--n-atoms-hist', str(args.n_atoms_hist.resolve()),
            '--marginal-dists', str(args.marginal_dists.resolve()), '--output-dir', str(update)], timeout=28800)
    child = update / 'last.ckpt'
    metrics = evaluate(args, config, frozen, output, child)
    immutable_json(output / 'campaign_result.json', {
        'method': 'flow_grpo_ode_to_sde_v1', 'status': 'completed',
        'training_reward': REWARD_ENDPOINT, 'final_model': str(child),
        'training_result': str(update / 'training_result.json'),
        'reward_metrics': metrics, 'held_out_evaluation_pending': True, 'provider_calls': 0,
    })

if __name__ == '__main__':
    main()
