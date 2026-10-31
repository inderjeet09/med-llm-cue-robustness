# Pre-registration: dated rules and the frozen plan

Every decision rule in this project was written down, dated and committed **before** the results it governs were seen. This file collects those rules verbatim from the research record: (1) the kill tests that decided whether the idea was worth pursuing, (2) the rule freezing the cue templates, and (3) the Phase 5 plan (arms, hypotheses H1–H4, decision rules) frozen before any test-set run. Later changes are dated amendments, never edits. D-numbers refer to `docs/DECISIONS.md`.

## 1. Kill tests (written 2026-10-17, before any kill-test result was seen)

### Pre-registered protocol (frozen; any change gets a new dated section below, not an edit)
Common: MedQA-US 4-option **dev** only (D006), 300 questions sampled with seed 0 (`src/data/medqa.sample`).
Decoding greedy / temperature 0, seed 0.

#### Test A (stage 1) — `scripts/kill_test_a.py`, Kaggle
- Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct (4-bit NF4). OpenBioLLM-8B added as a *non-decisive* extra to
  validate D008.
- Prompt: BiasMedQA letter-only (`biasmedqa_instruction_v1`); answer = argmax next-token probability over A–D.
- Conditions: clean; misleading cue × {false consensus, frequency, recency}; each × {no mitigation, education,
  few-shot (1 negative + 1 positive TRAIN demonstration)}; plus a helpful cue (control, non-decisive).
- **GO** per model if mean drop (clean − misleading, over the 3 biases) **≥ 10 points** AND pooled recovery of the
  best prompt mitigation per bias **< 50 %**. A is GO if it holds for both decisive models (if only one: report as
  mixed, decide at Checkpoint 1).
- Stage 2 (only if stage 1 GO): mini QLoRA, 2k train pairs, 1 epoch; GO if ≥ 50 % of the drop recovered with ≤ 2
  points clean loss. Script written after stage 1.

#### Test B — `scripts/kill_test_b.py`, Azure gpt-4.1-mini + Qwen2.5-7B on Kaggle
- 300 dev questions, excluding questions overlapping Handler et al.'s MedQA-NOTA set (1 excluded).
- Variants: original; NOTA (correct text → "None of the other answers"; NONE reply also counted correct);
  removed (correct option deleted, 3 options remain; gold = NONE).
- Conditions: cot (zero-shot CoT, Bedi), handler_base, handler_safety (best prompt baseline), gtv (ours).
- **GO** if NOTA drop under cot **≥ 20 points** AND gtv recovers **≥ 50 %** of it AND gtv costs **≤ 2 points** on
  originals. Decided per model; B is GO if it holds for gpt-4.1-mini and Qwen2.5-7B (mixed → Checkpoint 1).
- Secondary (non-decisive): removed-variant abstention, gtv vs handler_safety on all variants, unparsed rate.

#### Uncertainty
n = 300 → a ±5–6-point 95 % CI on a single accuracy; paired bootstrap CIs on drops are reported. A criterion met
only by its point estimate with a CI straddling the threshold is reported as "borderline", not a clean GO.

### Stage 2 pre-registration — Test A, mini QLoRA (written 2026-10-18, BEFORE any stage-2 run)

Decision at Checkpoint 1: run stage 2 on both decisive models with a **stricter bar** (must beat the
education prompt), because education already recovers 47 % (Qwen) / 67 % (Llama) of the drop.

**Training data (TRAIN split only).** 2,000 MedQA train questions (sample seed 1). Each gives one pair
(clean, cued). Cue = misleading (random wrong option) for 2/3 of pairs, helpful (correct option) for 1/3, chosen by a
deterministic hash. Bias template drawn by hash from the arm's template set.

**Arms (per model, 1 training seed — kill test, not a final result):**
| arm | templates in training | loss | role |
|---|---|---|---|
| `all7` | all 7 BiasMedQA templates (eval templates seen; eval QUESTIONS unseen: dev) | CE(clean) + CE(cued) + λ·KL | **decisive** |
| `heldout4` | self_diagnosis, confirmation, cultural, status_quo (false consensus, frequency, recency never seen) | same | secondary: generalisation |
| `sft_clean` | none — same 2,000 questions, clean only | CE(clean) | secondary: is it just MedQA SFT? |

Loss: CE = −log P(gold letter) at the answer position (full vocabulary, letter variants "A"/" A" pooled, as in
evaluation). KL = KL(stop-grad p_clean ‖ p_cued) over the renormalised A–D distribution, λ = 1. Applied to helpful
pairs too (the answer must not depend on the cue; gold-label CE anchors both).
QLoRA: NF4 4-bit, fp16 compute; LoRA r = 16, α = 32, dropout 0.05, all attention + MLP projections; AdamW lr 2e-4,
cosine, 5 % warm-up; 16 pairs per optimiser step; 1 epoch (125 steps); gradient checkpointing; train seed 0.
No hyperparameter search (kill test). Model revisions pinned to the stage-1 commits.

**Evaluation.** Same 300 dev questions (seed 0), same prompt (`biasmedqa_instruction_v1`) and letter-probability
argmax as stage 1. Conditions: clean; misleading × {false consensus, frequency, recency}; helpful × same.
Base reference = stage-1 records. Extra base job: education prompt on clean questions and on helpful cues (its cost).

**GO rule (per model, `all7` arm; all three must hold):**
1. Recovery ≥ 50 %: (mean misleading acc FT − mean misleading acc base) / (clean base − mean misleading base) ≥ 0.5.
2. Clean cost ≤ 2 pp: clean base − clean FT ≤ 2 (point estimate; CI reported).
3. **Stricter bar:** mean misleading acc FT > mean misleading acc with the education prompt (stage 1), with the 95 %
   paired bootstrap CI over questions (per-question mean over the 3 biases) excluding zero.
Stage 2 = **GO** if both models pass; one model → MIXED (decide at Checkpoint 1b); none → NO-GO for method A.
Reported but not decisive: helpful-cue accuracy (FT should not lose the benefit of correct hints), cue-follow rate,
letter mass, the `heldout4` and `sft_clean` arms with the same three numbers.
Analysis: `scripts/analyze_kill_a2.py` (combines the Kaggle outputs with stage-1 records).

Outcome: idea B (none-of-the-options) failed its rule and was dropped; idea A passed both stages. Data: `results/pilot/`.

## 2. Cue templates (frozen at the end of Phase 2, before any test-set evaluation)

- **Pre-registered rule:** templates are frozen from now on (before any test evaluation). Clinician feedback leads only to exclusion of invalid items and a reported validity rate, never to template edits.

## 3. Main experiments (Phase 5 plan)

Status: **FROZEN — approved at Checkpoint 3 on 2026-10-20 (plan text as of commit 335ce58).**

### Part B: Pre-registered Phase 5 plan

#### Amendments (dated; made before any Phase 5 result exists)
- **2026-10-20, B7.1 acceptance rule for prefix-shared evaluation.** Old: ≥ 99 % answer agreement with the standard
  path. Reason for the change: D019 measured that two runs of the *standard* path alone agree on only 97.3–99.2 % of
  answers on the T4 GPU (batch-composition noise), so 99 % could fail for reasons unrelated to prefix sharing. New:
  accepted if per-condition accuracy is within 1 pp of the standard path and answer agreement is ≥ 97.3 % (the D019
  standard-vs-standard floor). Reference = the gap1 standard-path records on the same 600 dev questions. Only arms whose
  conditions share one instruction use prefix sharing (base, generic warning, trained arms); oracle-warning and
  few-shot prompts differ per cue and use the standard path. This path difference (≤ 1 pp noise) is reported.
- **2026-10-20, schedule.** P5-0 covers Qwen and Llama only (prompt arms are expensive: few-shot prompts are long);
  Med42's dev prompt-arm selection moves to the start of P5-4. No change to any hypothesis or arm.
- **2026-10-20, few-shot on test.** Measured cost: few-shot prompts are ~3× longer, ~2.5 GPU-h per model on dev and ~5
  on test (the B8 estimate missed this). New: on test, few-shot is run only for models where it is the best prompt
  baseline by the frozen dev rule. Its dev results are reported for all models. Base, generic and oracle warnings run on
  test for every model. H1(b) (vs the best prompt baseline) is unaffected. Saves ~10–15 GPU-h.

#### B1. Method (unchanged from the kill test, D014)
Counterfactual-invariance fine-tuning (CIFT): QLoRA on pairs (clean, cued) built from MedQA **train**.
Loss = CE(clean → gold) + CE(cued → gold) + λ·KL(stopgrad p_clean ‖ p_cued) over A–D. λ = 1, LoRA r = 16, α = 32,
lr 2e-4 cosine, 1 epoch, 2,000 pairs (2/3 misleading, 1/3 helpful), 16 pairs per step. **No hyperparameter search for
any method.** Every trained method uses these same settings and the same 2,000 questions (pair seed 1).

#### B2. Hint split (the generalisation test)
- **Training cues:** the paper's 7 BiasMedQA templates (4 families).
- **Held-out cues:** our 8 new templates (4 families: authority, external anchor, automation, convenience). Never seen
  by any trained method. Chosen before seeing any trained model's results on them, and they are the harder cues
  (A1, finding 1).

#### B3. Models
Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Llama3-Med42-8B (4-bit NF4, pinned revisions). gpt-4.1-mini as an
untrained reference (test: clean, 15 misleading cues, generic and oracle warnings; ~$3–4).

#### B4. Arms
| arm | trained? | seeds | models |
|---|---|---|---|
| base (no mitigation) | no | – | 3 |
| oracle warning (paper's education text) | no | – | 3 |
| generic warning (ours, realistic) | no | – | 3 |
| few-shot (paper's method, 4-option) | no | – | 3 |
| SFT-clean (plain fine-tuning on the 2,000 clean questions) | yes | 3 | 3 |
| BCT (Chua et al.: cued prompt → the base model's own clean answer, no gold labels) | yes | 3 | Qwen, Llama |
| **CIFT (ours)** | yes | 3 | 3 |
| ablation: CIFT without KL (λ = 0) | yes | 3 | Llama |
| ablation: CIFT without helpful pairs (all misleading) | yes | 3 | Llama |
| optional, if quota allows: CIFT trained on all 15 cues (no held-out); BCT on Med42 | yes | 3 | Llama / Med42 |

"Best prompt baseline" per model = the prompt arm with the highest mean misleading-cue accuracy over all 15 cues on
**dev** (a rule fixed now; it is generous to the baselines).

#### B5. Evaluation (MedQA **test**, 1,273 questions, 4-option, letter scoring, fixed inference settings, D019)
Conditions: clean; misleading × 15 cues; helpful × 8 (one per family: frequency, recency, self_diagnosis, status_quo,
authority_senior, referral_note, cds_tool, convenience_time). Prompt arms: clean + 15 misleading.
Transfer (exploratory): MedQA-MultiTurnRobust (1,050 items, MIT, revision 8f345fc): clean vs its *misleading_context*
paragraph (a different, evidence-style manipulation).

#### B6. Hypotheses and decision rules (test split; two-level bootstrap over seeds × questions, D012)
- **H1 (primary, generalisation):** on the **held-out** cues, CIFT's mean misleading accuracy is higher than (a) base,
  (b) the best prompt baseline, (c) SFT-clean, (d) BCT. Claimed per comparison when the 95 % CI excludes 0, and as a
  general result when it holds for ≥ 2 of 3 models (BCT: both of its 2 models).
- **H2 (in-distribution):** the same on the 7 training cues (new questions).
- **H3 (no clean cost):** CIFT clean accuracy ≥ base − 2 pp (point estimate) and the CIFT − SFT-clean CI lower bound
  > −2 pp.
- **H4 (mechanism, Llama):** CIFT > CIFT-without-KL on held-out misleading accuracy (CI excludes 0). Helpful-pair effect:
  CIFT vs CIFT-without-helpful on helpful-cue gain (report direction; no claim unless the CI excludes 0).
- **Reported regardless of outcome:** residual drop (clean − misleading), helpful-cue gain for every arm (we expect
  fine-tuning to shrink it, kill test), per-family results, transfer results, mean ± std over seeds, GPU cost per method.
- **If H1 fails against SFT-clean or BCT** while H1(a, b) hold, the paper is reframed honestly: "fine-tuning beats
  prompting; our objective adds no reliable gain" (integrity rule 7).

#### B7. Engineering before Phase 5 runs (local + one small Kaggle check)
1. **Prefix-shared evaluation:** all 24 conditions of a question share the instruction and question text, so its
   key/value cache is computed once. Target: ~3× faster test evaluation. Accepted only if, on dev, it agrees with the
   standard path on ≥ 99 % of answers and per-condition accuracy is within 1 pp. Otherwise use the standard path.
2. Training script for all arms (CIFT / no-KL / no-helpful / all15 / SFT-clean / BCT) with in-job test + transfer
   evaluation (adapters are not downloaded).
3. Few-shot in the 4-option format (the paper's headers); generic warning (frozen).
4. Analysis: two-level bootstrap tables → `results/tables/`, figures → `figures/`.
5. Unit tests for the new pieces.

#### B8. Compute plan (estimates from measured job times; recalculated after the first batch)
Per trained run: training 1.5 h (CIFT, BCT) or 0.75 h (SFT) + test evaluation ~0.7 h (prefix-shared; ~2.2 h without)
+ transfer ~0.25 h.

| batch | contents | est. GPU-h |
|---|---|---|
| P5-0 | prefix-eval check + prompt arms on dev (3 models) | ~3 |
| P5-1 | baselines on test (base + 3 prompt arms × 3 models) + CIFT seed 0 (Qwen, Llama) | ~11 |
| P5-2 | CIFT seeds 1–2 (Qwen, Llama) + SFT-clean seeds 0–2 (Qwen, Llama) | ~20 |
| P5-3 | BCT seeds 0–2 (Qwen, Llama) | ~16 |
| P5-4 | Med42: CIFT + SFT-clean × 3 seeds | ~13 |
| P5-5 | ablations (Llama): no-KL, no-helpful × 3 seeds | ~15 |
| total | | **~78 GPU-h** (~3 Kaggle weeks); optional arms +15 |

Batches are ordered by importance. Each fits one week's quota and one 12-hour session (two GPUs in parallel).
API: gpt-4.1-mini test reference ~$3–4 (total stays under ~$12 of $25).

#### B9. Risks
- The CIFT − SFT margin (+5–8 pp in the kill test, 1 seed) may shrink with 3 seeds.
- BCT may match CIFT. Then the contribution is the medical evaluation + held-out families + analysis.
- Hint discounting: fine-tuning shrinks the benefit of correct hints. Reported as a limitation.
- Contamination: MedQA may be in pretraining. Our metric is a within-question drop, which is partly robust to it.
