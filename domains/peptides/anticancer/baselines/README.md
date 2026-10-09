# Peptide reward-access baselines used in the paper plot

This directory contains runnable, source-complete implementations of every
baseline represented in the anticancer main plots or table. The DPO
curriculum is in the parent task folder. No script imports code or
parameters from an EXP directory or from another Git commit.

## Method-to-code map

| Plot label | Frozen implementation |
| --- | --- |
| Base generator (PepDFM) | `acp_direct/prepare.py`, `unguided_p0` |
| ACP tilt | `acp_direct/prepare.py`, `pool_tilt_acp` |
| ACP guidance | `acp_direct/prepare.py`, `local_guidance_acp` |
| ACP+ hemolysis guidance | `acp_direct/prepare.py`, `local_guidance_acp_safe` |
| ACP-DPO | `acp_dpo_nft/baseline.py`, `dpo1` |
| DiffusionNFT | `acp_dpo_nft/baseline.py`, `nft11` |

The direct baselines are the exact EXP-0107 ACP-A implementations. ACP tilt
uses finite-pool sampling-importance-resampling from a 10x PepDFM pool with
weights proportional to ACP-A. The guidance arms use the released MOG-DFM
adaptive-hypercone discrete sampler. The safety arm optimizes
`log p(ACP-A) + log(1-p(hemolysis proxy))`.

ACP-DPO draws 100 base candidates, takes the 20 distinct highest and 20
distinct lowest ACP-A sequences, forms all 400 positive-by-negative pairs,
and performs one 100-step flow-DPO update (`beta=2.6`, batch size 32,
learning rate `1e-5`). DiffusionNFT is the explicitly documented discrete
adaptation of DiffusionNFT: 11 online rounds, 100 reward queries and 100
optimizer steps per round, batch size 64, `beta=1`, and a persistent optimizer.
It is an adaptation to categorical peptide flow matching, not a literal
reproduction of the continuous-image method.

All arms use three seeds and draw 100 peptides per replicate: 35 of length 12,
40 of length 24, and 25 of length 38. AntiCP2-alternate and HemoPI2 are applied
only after sampling. ACP-A is the reward for the baselines and therefore is a
mechanism check, not an independent validation endpoint.

## Runtime assets

Fetch PepDFM with `domains/peptides/fetch_assets.py`. Fetch the exact public
AntiCP2 and HemoPI2 revisions with:

```bash
python -m domains.peptides.fetch_predictors \
  --output-root /scratch/my-predictors
```

The script pins AntiCP2 commit `627105e...` and HemoPI2 commit `2b67a5c...`,
and verifies the two released AntiCP2 model hashes. Create the separate HemoPI2 compatibility overlay:

```bash
pip install --no-deps scikit-learn==1.3.1 joblib==1.3.2 --target /path/to/overlays/hemopi2
```

The baseline job wrappers also accept a torchdiffeq overlay; a normal peptide
installation can use its environment's site-packages directory for that argument.
Model weights and environments are runtime assets and are not committed.

## Slurm entry points

Submit through the site's tracked launcher from the repository root. The
direct runner takes:

```text
domains/peptides/anticancer/baselines/acp_direct/jobs/run.sh \
  OUTPUT PEPDFM_ASSETS PEPDFM_PYTHON HEMOPI2_MODEL HEMOPI2_OVERLAY \
  TORCHDIFFEQ_OVERLAY ANTICP2_ASSETS
```

It generates and scores Base, ACP tilt, ACP guidance and ACP+hemolysis
guidance. The DPO/NFT runner takes:

```text
domains/peptides/anticancer/baselines/acp_dpo_nft/jobs/run.sh \
  OUTPUT PEPDFM_ASSETS PEPDFM_PYTHON ANTICP2_ASSETS HEMOPI2_MODEL \
  HEMOPI2_OVERLAY TORCHDIFFEQ_OVERLAY
```

It independently trains ACP-DPO and DiffusionNFT from the released PepDFM
checkpoint and emits three-seed means and sample standard deviations for
ACP-A, AntiCP2-main, AntiCP2-alternate and HemoPI2. Neither baseline runner
needs an API key. Configuration, hashes, seeds and all heuristics are in the
adjacent `config.json` files.

The original source snapshots came from commits `41e34bf` (direct ACP-A
baselines) and `8e97e41` (ACP-DPO/DiffusionNFT); portability edits only remove
historical absolute paths and add the AntiCP2-alternate terminal metric needed
by the figure.
