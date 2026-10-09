# PepDFM anticancer DPO protocol

This fresh-run protocol starts from the released PepDFM checkpoint.

## Scientific question and fixed final goal

The immutable final goal `F` is:

> Prefer peptides that appear more promising as selective direct anticancer
> therapeutics: plausibly active against cancer cells while minimizing
> nonspecific toxicity toward healthy human cells.

The system tests whether iterative preference-based updates make progressively
tighter, scientifically related goals reachable by the generator. The intended
semantic relationship is

\[
\operatorname{Pos}(F) \subseteq \operatorname{Pos}(G_m)
\subseteq \cdots \subseteq \operatorname{Pos}(G_{02})
\subseteq \operatorname{Pos}(G_{01}).
\]

This is a sequence-judgeability and generator-reachability experiment. LLM
labels are not assays and cannot establish anticancer activity, selectivity,
toxicity, safety, or therapeutic efficacy.

## End-to-end state machine

1. Before seeing any sequence, the controller chooses two to four exact peptide
   lengths in `[6, 49]`, with 10--60 samples per chosen length and exactly 100
   samples in total. The allocation is frozen for the entire trajectory.
2. The released unconditional PepDFM checkpoint generates 100 unique valid
   peptides, forming `p0`. Target-independent predictor values remain private;
   the LLM sees only qualitative bands fitted once on `p0`.
3. The candidate judge evaluates exact `F` twice, using independently shuffled
   complete passes over the same 100 candidates.
4. The controller proposes a broad, weak, scientifically meaningful `G01`.
   A separate auditor checks semantic direction. The judge evaluates the
   proposal twice. A failed proposal may be materially corrected up to three
   times on the unchanged cohort.
5. Stable positive and stable negative examples from the accepted goal become
   the update buffer. Abstentions and disagreements are excluded.
6. One 100-step PepDFM update is performed with the DPO loss. A new
   100-peptide child distribution is sampled with the same length mixture.
7. The unchanged current goal is judged twice on the child. If it has not moved
   enough, the same goal is trained again. If it has moved enough, exact `F` is
   tested and, if still unavailable, the controller proposes the smallest
   newly reachable tightening.
8. The loop stops on success, an unproductive goal, failure to find an acceptable
   next goal, the semantic/update limit, a non-collapse failure, or a resource
   limit. The terminal validator reconstructs decisions from immutable raw
   artifacts rather than trusting the process exit code.

Every generated cohort is both the adaptive decision cohort and the training
cohort. There is no independent biological test set; this limitation is part of
the registered claim boundary.

## Stable judgments and the reported fraction

For candidate `i`, let the two categorical labels be
`y_i^(1), y_i^(2) in {positive, negative, abstain}`. The stable label is the
shared label when they agree and `unstable` otherwise. The reported quantity is

\[
q(G,p)=\frac{\#\{i:y_i^{(1)}=y_i^{(2)}=\mathrm{positive}\}}{100}.
\]

All 100 candidates remain in the denominator. Stable abstentions and unstable
judgments therefore lower `q`; they are never silently discarded. Training
uses only the stable-positive and stable-negative intersections.

## Fixed host gates and heuristics

| Condition | Frozen value |
| --- | ---: |
| Cohort size | 100 |
| Judgment passes per goal/distribution cell | 2 |
| Minimum stable positives | 20 |
| Minimum stable negatives | 20 |
| Minimum exact three-class pass agreement | 0.50 |
| Maximum abstain fraction in either pass | 0.50 |
| Minimum `G01 - F` stable-positive gain on `p0` | 0.10 |
| Current-goal prevalence before tightening | `q >= 0.60` |
| Current-goal enrichment from its entry distribution | `delta q >= 0.10` |
| Proposed tighter-goal unlock | `q_current - q_entry >= 0.10` |
| Proposal attempts | 1 initial + at most 3 corrections |
| Updates at one fixed goal | at most 5 |
| Total updates | at most 10 |
| Semantic tightenings | at most 3 |

The prevalence and delta conditions are conjunctive. For example, movement
from `0.42` to `0.55` clears the delta but not the `0.60` prevalence floor, so
the goal is held. These are absolute percentage-point differences, not relative
percentage changes.

Proposal acceptance also requires an auditor-approved semantic direction, the
unchanged length allocation, one witness that is stably positive, one witness
that is stably negative, and the learnability conditions above. Bootstrap
intervals are descriptive and do not replace the deterministic gates.

## Shared training and sampling parameters

| Parameter | Value |
| --- | ---: |
| Learning rate | `1e-5` |
| Update steps | 100 |
| Batch size | 32 |
| Gradient clipping | 0.1 |
| Time endpoint epsilon | 0.001 |
| Training seed | 8427 |
| Round seed stride | 1000 |
| Initial-pool seed | 8400 |
| Child-sampling seed | 8439 |
| Judge ordering seed | 8471 |
| Flow sampling steps | 100 |
| Sampling batch size | 60 |
| Source distribution | uniform |

The candidate judge is configured as `gemini-3.7-flash`, while the curriculum
controller and semantic auditor use `gemini-3.1-pro-preview`; all use high
reasoning effort, temperature zero in the client, no automatic retry, bounded
output and call budgets, and immutable request/response ledgers.

Child checkpoints must change state and their samples must pass all of these
non-collapse gates: 100% validity, 100% unique validity, zero exact replay of
the label buffer, zero overlap with prior distributions, at least 0.95 novelty
relative to `p0`, at least 0.90 diversity relative to `p0`, and amino-acid
composition Jensen--Shannon divergence no greater than 0.08.

## Flow-DPO update

Let `P` and `N` be the current cohort's distinct stable-positive and
stable-negative sequences. The DPO profile materializes every ordered
preference pair in the Cartesian product `P x N`; abstentions and unstable
labels are absent. Since the cohort has 100 items, the largest possible balanced
pair set is `50 x 50 = 2500`, below the fixed 3,200 training slots.

For preferred sequence `w`, rejected sequence `l`, trainable policy `pi`, and
the frozen parent checkpoint `ref`, the minimized loss is

\[
-\log\sigma\!\left[-\frac{\beta}{2}
\left(
[L_{\pi}(w)-L_{\pi}(l)]-[L_{\mathrm{ref}}(w)-L_{\mathrm{ref}}(l)]
\right)\right],\qquad \beta=2.6.
\]

`L` is PepDFM's per-sequence generalized-KL flow-matching surrogate. The
policy and reference see identical sampled corruptions for each sequence. A
deterministically shuffled epoch schedule must visit every pair before reuse,
and the parent reference is hash-checked to remain unchanged. This is a
flow/diffusion DPO surrogate, not exact sequence-likelihood DPO.

## Prompts and request construction

The runtime reads exact prompt text from [the shared peptide prompt file](../prompts.py).
The task-specific response schemas and request builders are in the adjacent
Python modules. Run records preserve the prompt and input bindings.

Requests, raw responses, parsed structured outputs, model-version bindings, and
usage accounting are persisted under `campaign/api_artifacts/` and
`campaign/api_attempts/`.

## Execution and validation

Use the fresh-run commands in [README.md](README.md). The job performs provider
preflight, length selection, asset validation, initial sampling and evidence
freezing, config binding, the adaptive DPO loop, and terminal validation.

A live run is terminal only when
`campaign/final_report.json` and `campaign/final_validation.json` agree with
the frozen configuration and artifact chain. Process completion alone is not
scientific success.
