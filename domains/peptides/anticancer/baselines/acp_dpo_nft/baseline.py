#!/usr/bin/env python3
"""EXP123: ACP-ranked flow-DPO and an explicitly discrete NFT adaptation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SCRATCH = Path(os.environ.get('LTFD_SCRATCH_ROOT',
    str(PROJECT / 'outputs')))
LENGTHS = {12: 35, 24: 40, 38: 25}
AA = 'ACDEFGHIKLMNPQRSTVWY'
ARMS = ('base', 'dpo1', 'nft1', 'nft11')
METRICS = ('ACP_A', 'AntiCP2_main', 'AntiCP2_alternate', 'HemoPI2')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    """Immutable publication; recovery may only reuse byte-equivalent JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f'immutable artifact differs: {path}')
        return
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('x') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, path)
    temp.unlink()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'input hash mismatch: {path}')
    return Path(path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def config():
    c = read(HERE / 'config.json')
    if c['seeds'] != [12301, 12302, 12303] or c['evaluation_seeds'] != [10701, 10702, 10703]:
        raise ValueError('unregistered seed change')
    c['output'] = os.environ.get('LTFD_BASELINE_OUTPUT', c['output'])
    c['asset_dir'] = os.environ.get('PEPDFM_ASSET_DIR', c['asset_dir'])
    c['anticp2_assets'] = os.environ.get('ANTICP2_ASSETS', c['anticp2_assets'])
    c['hemopi2_assets'] = os.environ.get('HEMOPI2_ASSETS', c['hemopi2_assets'])
    c['hemopi2_overlay'] = os.environ.get('HEMOPI2_OVERLAY', c['hemopi2_overlay'])
    for key in ('output', 'asset_dir', 'anticp2_assets', 'hemopi2_assets',
                'hemopi2_overlay', 'acp_parameters'):
        value = Path(c[key])
        if not value.is_absolute():
            c[key] = str((HERE / value).resolve())
    for path, digest in c['pinned_files'].items():
        checked(PROJECT / path, digest)
    for path, digest in c['asset_pinned_files'].items():
        checked(Path(c['asset_dir']) / path, digest)
    for path, digest in c['anticp2_pinned_files'].items():
        checked(Path(c['anticp2_assets']) / path, digest)
    return c


def acp_scores(rows, params):
    # Frozen EXP106 AAC20 + OVERLAPPING DPC400 feature convention.
    coef, mean, scale = (np.asarray(params[k]) for k in ('coefficients', 'mean', 'scale'))
    out = []
    for row in rows:
        s = row['sequence']
        if not row['is_valid'] or not s or not set(s) <= set(AA):
            raise ValueError('invalid training draw; preserve and fail, never replace')
        single = Counter(s)
        pairs = Counter(s[i:i+2] for i in range(len(s)-1))
        features = np.asarray([single[a]/len(s) for a in AA] +
                              [pairs[a+b]/(len(s)-1) for a in AA for b in AA])
        z = float(((features-mean)/scale) @ coef + params['intercept'])
        out.append(float(np.exp(-np.logaddexp(0., -z))))
    return np.asarray(out)


def extremes(rows, rewards):
    unique = {}
    for row, reward in zip(rows, rewards):
        unique.setdefault(row['sequence'], float(reward))
    ranked = sorted(unique, key=lambda s: (unique[s], s))
    if len(ranked) < 40:
        raise ValueError('need at least40 distinct canonical training sequences')
    if unique[ranked[19]] == unique[ranked[20]] or unique[ranked[-21]] == unique[ranked[-20]]:
        raise ValueError('ACP tie at top/bottom cutoff; no arbitrary preference split')
    neg, pos = ranked[:20], ranked[-20:]
    if set(pos) & set(neg) or min(unique[s] for s in pos) <= max(unique[s] for s in neg):
        raise ValueError('strict disjoint ACP preference ordering required')
    return pos, neg


def optimality(rewards, lengths):
    rewards, lengths = np.asarray(rewards, float), np.asarray(lengths)
    std = float(rewards.std(ddof=0))
    result = np.full(len(rewards), .5)
    if std > 1e-8:
        for length in np.unique(lengths):
            group = lengths == length
            result[group] += .5*np.clip((rewards[group]-rewards[group].mean())/std, -1, 1)
    return result


def nft_loss(student, old, target, reward, beta=1., epsilon=1e-5):
    """Paper Eq5/§3.3, adapted to categorical clean-posterior regression.

    Affine branches are regression vectors, NOT probabilities. Never clamp or
    softmax the negative branch; do not apply a categorical KL to it.
    Reduction: each sequence over ALL encoded positions and 24 categories.
    """
    old = old.detach()
    plus = (1-beta)*old + beta*student
    minus = (1+beta)*old - beta*student
    dims = tuple(range(1, student.ndim))
    def branch(pred):
        error = pred-target
        denominator = error.abs().mean(dims).detach().clamp_min(epsilon)
        return error.square().mean(dims)/denominator
    return reward*branch(plus) + (1-reward)*branch(minus)


def draw(model, seed, label, c):
    rows = []
    for length, count in LENGTHS.items():
        sampling_seed = seed + length*1000
        samples = model.sample(num_samples=count, length=length, num_steps=c['sampling_steps'],
                               seed=sampling_seed, source_distribution='uniform')
        rows.extend({**asdict(s), 'requested_length': length, 'sampling_seed': sampling_seed,
                     'draw_id': f'{label}-L{length}-{s.sample_index:03d}'} for s in samples)
    if len(rows) != 100:
        raise ValueError('draw count differs')
    return rows


def evaluate_draw(model, root, arm, seed, eval_seed, c, checkpoint_hash):
    path = root / 'evaluation' / f'{arm}.json'
    if path.exists():
        value = read(path)
        if value['checkpoint_sha256'] != checkpoint_hash or value['config_sha256'] != sha(HERE/'config.json'):
            raise ValueError('evaluation checkpoint/config mismatch')
        return
    rows = draw(model, eval_seed, f'{seed}-{arm}', c)
    historical_matches = None
    if arm == 'base' and c.get('historical_base_root'):
        reference = Path(c['historical_base_root'])/f'replicate-{c["seeds"].index(seed)}'
        historical = sum((read(reference/f'raw_unguided_p0_L{length}.json') for length in LENGTHS), [])
        historical_matches = ([r['token_ids'] for r in historical] == [list(r['token_ids']) for r in rows])
    write(path, {'arm': arm, 'seed': seed, 'evaluation_seed': eval_seed,
                 'checkpoint_sha256': checkpoint_hash, 'config_sha256': sha(HERE/'config.json'),
                 'exact_historical_base_draw_match': historical_matches, 'rows': rows})


def attempt_path(root):
    root.mkdir(parents=True, exist_ok=True)
    candidate = root/f'attempt-{os.environ["SLURM_JOB_ID"]}'
    candidate.mkdir(exist_ok=False)
    return candidate


def train_nft_round(policy, old, optimizer, rows, rewards, seed, c):
    import torch
    from domains.peptides.checkpoint import sha256_state_dict
    old_before = sha256_state_dict(old.model)
    old.model.eval()
    policy.model.train()
    r = optimality(rewards, [r['requested_length'] for r in rows])
    # Every rollout draw retains its multiplicity; one shuffled stream cycles.
    order = []
    rng = random.Random(seed)
    while len(order) < c['steps_per_update']*c['nft']['batch_size']:
        epoch = list(range(len(rows)))
        rng.shuffle(epoch)
        order.extend(epoch)
    losses, gradnorms = [], []
    with torch.random.fork_rng(devices=[policy.device]):
        torch.manual_seed(seed)
        for step in range(c['steps_per_update']):
            batch = order[step*c['nft']['batch_size']:(step+1)*c['nft']['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            for length in LENGTHS:
                group = [i for i in batch if rows[i]['requested_length'] == length]
                if not group:
                    continue
                x1 = policy._tokens([rows[i]['sequence'] for i in group])
                x0 = torch.randint(0, 24, x1.shape, device=policy.device)
                t = torch.rand(len(group), device=policy.device)*(1-c['epsilon'])
                sample = policy.path.sample(t=t, x_0=x0, x_1=x1)
                q = policy.model(sample.x_t, sample.t).softmax(-1)
                with torch.no_grad():
                    qold = old.model(sample.x_t, sample.t).softmax(-1)
                target = torch.nn.functional.one_hot(x1, 24).to(q.dtype)
                reward = torch.tensor(r[group], device=policy.device, dtype=q.dtype)
                loss = nft_loss(q, qold, target, reward, c['nft']['beta']).sum()/len(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite NFT loss')
                loss.backward()
                loss_value += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(policy.model.parameters(), c['gradient_clip'])
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite NFT gradient')
            optimizer.step()
            losses.append(loss_value)
            gradnorms.append(float(norm))
        rng_state = {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(policy.device)}
    if sha256_state_dict(old.model) != old_before or any(p.grad is not None for p in old.model.parameters()):
        raise RuntimeError('frozen NFT old policy changed or acquired gradients')
    policy.model.eval()
    return {'losses': losses, 'preclip_gradient_norms': gradnorms, 'optimality': r.tolist(),
            'old_state_hash': old_before, 'sequence_exposures': len(batch)*c['steps_per_update']}, rng_state


def run_seed(c, seed):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('real generation/training requires tracked GPU Slurm job')
    import torch
    sys.path.insert(0, str(PROJECT))
    from domains.peptides.model import PepDfmAssets
    from domains.peptides.checkpoint import PepDfmCheckpoint, sha256_state_dict
    from domains.peptides.dpo import FlowDPOConfig, fine_tune_flow_dpo, write_or_validate_pair_manifest
    root = Path(c['output'])/str(seed)
    root.mkdir(parents=True, exist_ok=True)
    lock = (root/'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(root/'protocol.json', {'config_sha256':sha(HERE/'config.json'), 'seed':seed})
    if (root/'completion.json').exists():
        checked(HERE/'config.json', read(root/'completion.json')['config_sha256'])
        print(f'{seed}: already complete', flush=True)
        return
    if not torch.cuda.is_available() or torch.__version__.split('+')[0] != '2.6.0':
        raise RuntimeError('unexpected GPU runtime')
    attempt = attempt_path(root/'worker_attempts')
    write(attempt/'environment.json', {'python': sys.version, 'torch': torch.__version__,
          'numpy': np.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
          'config_sha256': sha(HERE/'config.json'), 'code_commit': subprocess.check_output(
              ['git','rev-parse','HEAD'], cwd=HERE, text=True).strip(),
          'job_id': os.environ['SLURM_JOB_ID'], 'seed': seed})
    freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    write(attempt/'packages.json', {'pip_freeze':freeze,'pip_freeze_sha256':hashlib.sha256(freeze.encode()).hexdigest()})
    assets = PepDfmAssets.resolve(Path(c['asset_dir']))
    base = PepDfmCheckpoint(assets, device='cuda:0')
    eval_seed = c['evaluation_seeds'][c['seeds'].index(seed)]
    evaluate_draw(base, root, 'base', seed, eval_seed, c, sha(assets.checkpoint))
    initial = root/'initial_training_draws.json'
    if not initial.exists():
        write(initial, draw(base, 1_000_000+seed, f'{seed}-train-initial', c))
    rows = read(initial)
    params = read(c['acp_parameters'])
    statuses = {}
    start = time.monotonic()
    try:
        rewards = acp_scores(rows, params)
        positive, negative = extremes(rows, rewards)
        write(root/'dpo_selection.json', {'positive':positive, 'negative':negative,
              'scored_draws':[{**row,'ACP_A':float(r)} for row,r in zip(rows,rewards)],
              'unique_count':len({r['sequence'] for r in rows}), 'ACP_queries':100,
              'rule':'global top20 and bottom20 distinct sequences; ties at cutoff fail'})
        pairs = root/'dpo_pairs.jsonl'
        write_or_validate_pair_manifest(pairs, positive_sequences=positive, negative_sequences=negative)
        policy = base.clone()
        receipt = root/'dpo_complete.json'
        if receipt.exists():
            meta = read(receipt)
            policy.load_checkpoint(Path(meta['checkpoint']), expected_sha256=meta['checkpoint_sha256'])
        else:
            directory = attempt_path(root/'dpo')
            trace = fine_tune_flow_dpo(policy=policy, reference=base, positive_sequences=positive,
                negative_sequences=negative, pair_manifest_path=pairs,
                config=FlowDPOConfig(learning_rate=c['learning_rate'], steps=c['steps_per_update'],
                                     batch_size=c['dpo']['batch_size'], beta=c['dpo']['beta'],
                                     gradient_clip_norm=c['gradient_clip'], seed=3_000_000+seed))
            checkpoint = directory/'student.pt'
            digest = policy.save_checkpoint(checkpoint)
            write(directory/'training.json', trace)
            meta = {'checkpoint':str(checkpoint), 'checkpoint_sha256':digest,
                    'training_sha256':sha(directory/'training.json'), 'ACP_queries':100,
                    'optimizer_steps':100, 'elapsed_seconds':time.monotonic()-start}
            write(receipt, meta)
        evaluate_draw(policy, root, 'dpo1', seed, eval_seed, c, meta['checkpoint_sha256'])
        statuses['dpo1'] = 'complete'
        del policy
    except Exception:
        statuses['dpo1'] = 'failed'
        write(attempt/'dpo_failure.json', {'traceback':traceback.format_exc()})
        traceback.print_exc()
    print(f'{seed}: DPO {statuses["dpo1"]}', flush=True)
    policy, old = base.clone(), base.clone()
    del base
    for p in old.model.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.model.parameters(), lr=c['learning_rate'])
    try:
        for iteration in range(1, c['nft']['rounds']+1):
            iteration_start = time.monotonic()
            receipt = root/'nft'/f'round-{iteration:02d}'/'completion.json'
            if receipt.exists():
                meta = read(receipt)
                state = torch.load(checked(meta['checkpoint'],meta['checkpoint_sha256']),
                                   weights_only=True, map_location=policy.device)
                policy.model.load_state_dict(state['student'], strict=True)
                old.model.load_state_dict(state['old'], strict=True)
                optimizer.load_state_dict(state['optimizer'])
                if state['iteration'] != iteration or sha256_state_dict(policy.model) != meta['student_state_hash']:
                    raise ValueError('NFT continuation state identity differs')
            else:
                directory = attempt_path(receipt.parent)
                rollout = rows if iteration == 1 else draw(old, 1_000_000+seed+(iteration-1)*100_000,
                                                          f'{seed}-nft-rollout-{iteration}', c)
                write(directory/'rollout.json', rollout)
                rewards = acp_scores(rollout, params)
                trace, rng_state = train_nft_round(policy, old, optimizer, rollout, rewards,
                                                  4_000_000+seed+iteration*100_000, c)
                eta = min(.001*iteration, .5)
                with torch.no_grad():
                    for old_p, new_p in zip(old.model.parameters(),policy.model.parameters()):
                        old_p.mul_(eta).add_(new_p, alpha=1-eta)
                trace.update(raw_ACP=rewards.tolist(), ema_decay=eta,
                             old_state_hash_after_ema=sha256_state_dict(old.model))
                write(directory/'training.json', trace)
                checkpoint = directory/'state.pt'
                torch.save({'student':policy.model.state_dict(), 'old':old.model.state_dict(),
                            'optimizer':optimizer.state_dict(), 'rng_state':rng_state,
                            'iteration':iteration}, checkpoint)
                meta = {'checkpoint':str(checkpoint), 'checkpoint_sha256':sha(checkpoint),
                        'student_state_hash':sha256_state_dict(policy.model), 'iteration':iteration,
                        'training_sha256':sha(directory/'training.json'),
                        'rollout_sha256':sha(directory/'rollout.json'), 'ACP_queries_cumulative':iteration*100,
                        'optimizer_steps_cumulative':iteration*c['steps_per_update'],
                        'elapsed_seconds':time.monotonic()-iteration_start}
                write(receipt,meta)
            if iteration in (1,c['nft']['rounds']):
                evaluate_draw(policy, root, f'nft{iteration}', seed, eval_seed, c, meta['student_state_hash'])
                statuses[f'nft{iteration}'] = 'complete'
            print(f'{seed}: NFT round {iteration}/{c["nft"]["rounds"]} complete', flush=True)
    except Exception:
        statuses['nft11'] = 'failed'
        write(attempt/'nft_failure.json', {'traceback':traceback.format_exc()})
        traceback.print_exc()
    final = {'seed':seed,'statuses':statuses,'all_complete':all(statuses.get(a)=='complete' for a in ARMS[1:]),
             'config_sha256':sha(HERE/'config.json')}
    write(attempt/'outcome.json', final)
    if final['all_complete']:
        write(root/'completion.json',final)
    print(json.dumps(final),flush=True)


def summarize_cohort(rows):
    result = {'draw_count':len(rows),'valid_count':sum(r['is_valid'] for r in rows),'by_length':{}}
    for length,count in LENGTHS.items():
        group = [r for r in rows if r['requested_length']==length]
        if len(group)!=count:
            raise ValueError('fixed length draw allocation changed')
        valid = [r for r in group if r['is_valid']]
        values = {m:float(np.mean([r[m] for r in valid])) if valid else None for m in METRICS}
        seqs = [r['sequence'] for r in valid]
        pairs = len(seqs)*(len(seqs)-1)//2
        hamming = (sum(pairs-sum(n*(n-1)//2 for n in Counter(position).values())
                        for position in zip(*seqs))/(pairs*length)) if pairs else None
        result['by_length'][str(length)] = {**values,'draw_count':count,'valid_count':len(valid),
                    'unique_count':len(set(seqs)), 'pairwise_hamming':hamming}
    for metric in (*METRICS,'pairwise_hamming'):
        result[metric] = (sum(count/100*result['by_length'][str(length)][metric]
                         for length,count in LENGTHS.items()) if all(
                         result['by_length'][str(length)][metric] is not None for length in LENGTHS) else None)
    result['unique_fraction_all_draws'] = len({r['sequence'] for r in rows if r['is_valid']})/100
    return result


def aggregate(cohorts, seeds):
    result = {}
    for arm in ARMS:
        present = {seed:cohorts.get(f'{seed}-{arm}') for seed in seeds}
        item = {'planned_seeds':seeds,'available_seeds':[s for s,v in present.items() if v],
                'missing_seeds':[s for s,v in present.items() if not v], 'replicates':present}
        for m in METRICS:
            vals = [present[s][m] for s in seeds if present[s] and present[s][m] is not None]
            # Never disguise survivor-only statistics as the preregistered result.
            item[m] = {'mean':float(np.mean(vals)) if len(vals)==len(seeds) else None,
                       'sd':float(np.std(vals,ddof=1)) if len(vals)==len(seeds) else None,
                       'defined_n':len(vals)}
            if len(vals)==len(seeds) and all(cohorts.get(f'{s}-base',{}).get(m) is not None for s in seeds):
                delta = np.array(vals)-np.array([cohorts[f'{s}-base'][m] for s in seeds])
                item[m].update(paired_base_change_mean=float(delta.mean()),
                               paired_base_change_sd=float(delta.std(ddof=1)))
        result[arm]=item
    return result


def report(c):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('full predictor evaluation requires tracked Slurm job')
    root = Path(c['output'])
    directory = attempt_path(root/'reports')
    legacy = module(PROJECT/'domains/peptides/anticancer/baselines/evaluation/anticp2_frozen.py','legacy_anticp2')
    assets = Path(c['anticp2_assets'])
    main = legacy.load_legacy_forest(assets/'dpc_extra_model',400)
    alternate = legacy.load_legacy_forest(assets/'aac_extra_model',20)
    if list(main.classes_) != [-1,1] or list(alternate.classes_) != [-1,1]:
        raise ValueError('AntiCP2 class order changed')
    if not np.isclose(legacy.legacy_predict_proba(main,legacy.dpc_features(['APMGFQGMRG']))[0,1],.53,atol=1e-12,rtol=0):
        raise ValueError('released AntiCP2 example differs')
    all_rows, inputs, sources = [],{},{}
    for seed in c['seeds']:
        for arm in ARMS:
            path = root/str(seed)/'evaluation'/f'{arm}.json'
            if path.exists():
                data = read(path)
                checked(HERE/'config.json',data['config_sha256'])
                inputs[str(path)]=sha(path)
                sources[f'{seed}-{arm}']=data
                all_rows.extend({**row,'seed':seed,'arm':arm} for row in data['rows'])
    valid = [r for r in all_rows if r['is_valid']]
    if len({r['draw_id'] for r in all_rows})!=len(all_rows):
        raise ValueError('duplicate draw IDs')
    params = read(c['acp_parameters'])
    acp = acp_scores(valid,params)
    seqs = [r['sequence'] for r in valid]
    pm = legacy.legacy_predict_proba(main,legacy.dpc_features(seqs))[:,1]
    pa = legacy.legacy_predict_proba(alternate,legacy.aac_features(seqs))[:,1]
    fasta = directory/'candidates.fasta'
    with fasta.open('x') as f:
        f.write(''.join(f'>{r["draw_id"]}\n{r["sequence"]}\n' for r in valid))
    environment = dict(os.environ)
    environment.update(PYTHONPATH=str(c['hemopi2_overlay']),
                       PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    environment.pop('GEMINI_API_KEY',None)
    environment.pop('GOOGLE_API_KEY',None)
    hemo = directory/'hemopi2'
    with (directory/'hemopi2.log').open('x') as log:
        subprocess.run([sys.executable,str(PROJECT/'domains/peptides/anticancer/baselines/evaluation/hemopi2_rf_adapter.py'),
             '--input-fasta',str(fasta),'--assets',str(c['hemopi2_assets']),
             '--output',str(hemo),'--mode','classification','--threshold','0.45','--allow-model-load'],
             env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    receipt = read(hemo/'completion.json')
    if receipt.get('complete') is not True or receipt['input_sha256']!=sha(fasta):
        raise ValueError('HemoPI2 provenance mismatch')
    predictions = read(checked(hemo/'predictions.json',receipt['predictions_sha256']))
    indexed = {r['row_id']:r for r in predictions}
    if len(predictions)!=len(valid) or len(indexed)!=len(valid) or set(indexed)!={r['draw_id'] for r in valid}:
        raise ValueError('HemoPI2 coverage mismatch')
    for row,a,m,alt in zip(valid,acp,pm,pa):
        h = indexed[row['draw_id']]
        if h['sequence']!=row['sequence'] or h['status']!='eligible':
            raise ValueError('HemoPI2 identity mismatch')
        row.update(ACP_A=float(a),AntiCP2_main=float(m),AntiCP2_alternate=float(alt),
                   HemoPI2=float(h['hemolysis_probability_raw']))
    cohorts = {key:summarize_cohort([r for r in all_rows if f'{r["seed"]}-{r["arm"]}'==key]) for key in sources}
    overlaps = {}
    for seed in c['seeds']:
        training_paths = [root/str(seed)/'initial_training_draws.json']+list((root/str(seed)/'nft').glob('round-*/attempt-*/rollout.json'))
        trainseqs = {r['sequence'] for path in training_paths if path.exists() for r in read(path) if r['is_valid']}
        overlaps[str(seed)]={arm:sum(r['is_valid'] and r['sequence'] in trainseqs for r in all_rows
                                   if r['seed']==seed and r['arm']==arm) for arm in ARMS}
    summary = {'experiment_id':'EXP-0123','cohorts':cohorts,'aggregate':aggregate(cohorts,c['seeds']),
        'input_sha256':inputs,'config_sha256':sha(HERE/'config.json'),'train_eval_overlap_counts':overlaps,
        'historical_base_exact_matches':{k:v['exact_historical_base_draw_match'] for k,v in sources.items() if v['arm']=='base'},
        'all_draws_preserved':True,'all_seed_arms_available':len(cohorts)==12,
        'claim_boundary':'ACP-A is optimized reward, not validation. AntiCP2 shares ACP training-source families. NFT is a discrete adaptation, not a literal paper reproduction. Historical selected curriculum is descriptive, not randomized replicated evidence.'}
    write(directory/'candidates.json',all_rows)
    write(directory/'report.json',summary)
    lines = ['# EXP123: ACP reward-access training baselines','',
             '| Method | ACP-A ↑ | AntiCP2-main ↑ | AntiCP2-alternate ↑ | HemoPI2 ↓ |',
             '|---|---:|---:|---:|---:|']
    labels={'base':'Regenerated matched base','dpo1':'ACP top20 × bottom20 DPO, one update',
            'nft1':'Discrete NFT adaptation, one update','nft11':'Discrete NFT adaptation, 11 updates'}
    for arm in ARMS:
        vals = summary['aggregate'][arm]
        cells=[f'{vals[m]["mean"]:.4f} ± {vals[m]["sd"]:.4f}' if vals[m]['mean'] is not None else 'incomplete' for m in METRICS]
        lines.append('| '+labels[arm]+' | '+' | '.join(cells)+' |')
    lines += ['', 'New rows: mean ± sample SD over all three planned training seeds (base: three sampling streams).',
        'All cohorts: 100 draws, lengths 12/24/38 with counts 35/40/25; fixed-length conditional means if invalids occur. See JSON for validity, failures, each seed, diversity, overlap, and paired base changes.',
        'DPO: 100 ACP queries, 100 steps, 6,400 sequence exposures. NFT@1: same counts; NFT@11: 1,100 queries, 1,100 steps, 70,400 exposures. No equal runtime, hyperparameter tuning, or theorem claim.',
        'Historical curriculum is one post-hoc selected trajectory; historical baselines average three sampling seeds, not three training runs.',
        'ACP-A is the optimized reward. AntiCP2-main shares the ACP training data source. Predictor scores are not biological efficacy/safety measurements.']
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n')
    write(directory/'completion.json',{'complete':True,'report_sha256':sha(directory/'report.json'),
          'candidates_sha256':sha(directory/'candidates.json'),'all_seed_arms_available':summary['all_seed_arms_available']})
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','report','audit'])
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    c=config()
    if args.action=='train':
        if args.seed not in c['seeds']:
            raise ValueError('unregistered seed')
        run_seed(c,args.seed)
    elif args.action=='report':
        report(c)
    else:
        print(json.dumps({'pins_verified':len(c['pinned_files']),'config_sha256':sha(HERE/'config.json')}))


if __name__=='__main__':
    main()
