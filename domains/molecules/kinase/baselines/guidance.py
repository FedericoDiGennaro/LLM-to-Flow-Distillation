#!/usr/bin/env python3
"""Fit a time-conditioned graph reward predictor and guide a frozen FlowMol model."""
from __future__ import annotations
from domains.molecules.kinase.artifacts import immutable_json
from domains.molecules.kinase.oracle import REWARD_ENDPOINT
from domains.molecules.kinase.run import _run
from domains.molecules.kinase.baselines.grpo import parser, initialize, collect_and_score, no_signal
from domains.molecules.kinase.baselines.dpo import score_pool, mean_attempt_reward

def main():
    args = parser().parse_args()
    config, output, frozen = initialize(args)
    if config['algorithm'] != 'time_conditioned_soft_graph_reward_guidance_v1':
        raise ValueError('expected a time-conditioned graph guidance profile')
    rollout, reward, data = collect_and_score(args, config, output)
    rewards = {float(row['raw_reward']) for row in data['samples']}
    if 0.0 not in rewards or not any(value > 0 for value in rewards):
        no_signal(output, args, config['algorithm'], data, status='no_update_insufficient_reward_strata')
        return
    training = output / 'predictor'
    predictor = training / 'reward_predictor.pt'
    if not predictor.exists():
        _run([str(args.flowmol_python.absolute()), '-m', 'domains.molecules.kinase.train_surrogate',
            '--config', str(frozen), '--trajectory', str(rollout / 'trajectories.pt'),
            '--samples', str(rollout / 'samples.pt'), '--reward', str(reward),
            '--parent-checkpoint', str(args.checkpoint.resolve()),
            '--n-atoms-hist', str(args.n_atoms_hist.resolve()),
            '--marginal-dists', str(args.marginal_dists.resolve()), '--output-dir', str(training)], timeout=28800)
    generation = output / 'development/generation'
    pool = output / 'development/pool'
    spec = config['fresh_sampling']
    if not (generation / 'samples.pt').exists():
        _run([str(args.flowmol_python.absolute()), '-m', 'domains.molecules.kinase.sample_guided',
            '--config', str(frozen), '--checkpoint', str(args.checkpoint.resolve()),
            '--predictor', str(predictor), '--n-atoms-hist', str(args.n_atoms_hist.resolve()),
            '--marginal-dists', str(args.marginal_dists.resolve()), '--output-dir', str(generation),
            '--count', str(spec['num_attempts_per_arm']), '--batch-size', str(spec['batch_size']),
            '--seed', str(spec['round_1_seeds'][0]), '--n-timesteps', str(spec['flow_timesteps'])], timeout=28800)
    if not (pool / 'pool_manifest.json').exists():
        _run([str(args.flowmol_python.absolute()), '-m', 'domains.molecules.kinase.pool',
            '--samples', str(generation / 'samples.pt'), '--config', str(frozen),
            '--output-dir', str(pool), '--python', str(args.flowmol_python.absolute()),
            '--role', 'development'], timeout=28800)
    scored = score_pool(tdc_python=args.tdc_python, pool=pool / 'candidate_pool.jsonl',
        oracle=args.oracle, output=output / 'evaluation', config=config, induce_preferences=False)
    immutable_json(output / 'campaign_result.json', {
        'method': config['algorithm'], 'status': 'completed',
        'training_reward': REWARD_ENDPOINT, 'final_model': str(args.checkpoint.resolve()),
        'predictor': str(predictor), 'reward_metrics': mean_attempt_reward(scored),
        'generator_updates': 0, 'held_out_evaluation_pending': True, 'provider_calls': 0,
    })

if __name__ == '__main__':
    main()
