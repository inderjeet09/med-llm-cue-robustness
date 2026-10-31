# docs/DECISIONS.md

Format: **Dxxx — title** · Decision (entries are in the order they were made) · Alternatives considered · Reasoning.

**D001 — Separate project folder and conda env**
- Decision: a new project folder, separate from the earlier ECG project; env `medllm` (Python 3.11).
- Alternatives: reuse ecg-research repo/env.
- Reasoning: the protocol forbids writing to the old project; a fresh env avoids old pins.

**D002 — All packages from conda-forge only (`--override-channels -c conda-forge`)**
- Decision: PyTorch 2.14.1 and every other package come from conda-forge; one OpenMP runtime (`llvm-openmp`).
- Alternatives: pip PyTorch wheels; defaults channel.
- Reasoning: last project hit an OpenMP duplicate-library crash from mixing pip/defaults PyTorch with conda-forge
  numpy. `tiktoken` added (token counting for cost estimates). `bitsandbytes` deliberately not installed on Mac.

**D003 — Price source and conservative default**
- Decision: prices in `configs/pricing.yaml` come from the official Azure Retail Prices API (the human pricing page
  renders prices with JavaScript, so a fetch showed only placeholders). Until the deployment
  type/region, the client bills at the highest listed standard rate ($0.55 / $0.138 / $2.20 per 1M input / cached /
  output) and records `price_basis=conservative` on each ledger row.
- Alternatives: assume Global Standard ($0.40 / $0.10 / $1.60).
- Reasoning: never guess a price; an over-estimate is safe for budgeting, an under-estimate is not.

**D004 — Response cache as one JSON file per request**
- Decision: `results/costs/api_cache/<k[:2]>/<sha256>.json`, gitignored.
- Alternatives: `diskcache` (SQLite).
- Reasoning: human-readable, easy to inspect an individual model answer, trivially portable; `diskcache` stays
  installed for other uses. Content-filter refusals and API errors are logged but not cached.

**D005 — Base papers**
- Decision: primary base paper Schmidgall et al. 2024 (BiasMedQA, axis A). Axis B base: Bedi et al. 2025 for the
  NOTA transformation, Handler et al. 2026 for the removed-answer protocol and its prompt baseline.
- Alternatives: Handler as primary (highest score in our candidate scoring, MIT code) — but it is an unrefereed
  preprint on frontier API models; Xiao R-FT — no code, multi-turn, already near-zero flip rates (little room).
- Reasoning: BiasMedQA is peer-reviewed, has a clear taxonomy and explicit prompt-only mitigations that "did not fully
  restore" accuracy, and runs on open models. Using two axes keeps a fallback if one fails its kill test.

**D006 — Question format: MedQA US 4-option, official splits, loaded from bigbio/med_qa**
- Decision: 4-option (`med_qa_en_4options_source`: 10,178/1,272/1,273). A faithful 5-option run of the BiasMedQA
  protocol is done only in Phase 3 for the paper-vs-ours comparison.
- Alternatives: 5-option everywhere (matches BiasMedQA).
- Reasoning: 4-option is the community standard, matches MultiTurnRobust and the dev split we need for kill tests and
  tuning; bigbio is the only mirror with all three official splits at the exact counts.

**D007 — Held-out bias split by family; BCT is a required baseline**
- Decision: generalisation tests hold out entire bias *families* (e.g. social-proof, authority, anchoring), so
  paraphrases of a training cue never count as "unseen". BCT (Chua et al. 2024) and plain SFT are required baselines
  for method A.
- Reasoning: Xiao et al.'s "unseen" = paraphrases, which overstates generalisation; BCT is the obvious reviewer
  question.

**D008 — Medical model: Llama3-OpenBioLLM-8B replaces BioMistral-7B (provisional)**
- Decision: aaditya/Llama3-OpenBioLLM-8B (Llama 3 licence, not gated), with the Llama-3-instruct chat template
  applied manually.
- Alternatives: BioMistral-7B (base model, 2048 ctx — poor fit for chat prompts); MedGemma-4B-it (newer, gated
  click-through, but ~4B so not size-matched).
- Reasoning: instruction-tuned, size-matched to the 7–8B general models. Revisit if its MedQA accuracy turns out
  unusable in the kill test.

**D009 — Scoring for no-correct-option variants**
- NOTA variant: a reply naming the NOTA letter OR stating NONE is correct (both say "none of the others is right").
  Removed variant: only NONE (abstain / none correct) is correct; with 4-option MedQA, 3 options remain (Handler had
  4 left because they started from 5-option items) — noted as a protocol difference.
- Unparseable replies count as wrong and are reported separately.

**D010 — Kaggle inference via HF transformers + bitsandbytes NF4 (not vLLM)**
- Alternatives: vLLM (much faster generation; but fp16 7B doesn't fit one T4, AWQ checkpoints differ from the
  checkpoints we fine-tune, and vLLM on T4 has had compatibility churn).
- Reasoning: the same loading path is used for inference and QLoRA training, so before/after comparisons use the
  identical quantised model. Revisit if generation speed blocks Phase 5.

**D011 — Handler's MedQA-NOTA items come from the MedQA TEST split**
- Finding: 488/500 of Handler's stems match our test split (80-char normalised prefix); 1 matches dev, 11 train
  (likely shared openings / near-duplicates). Their paper says "validation".
- Decision: exclude overlapping dev items from dev experiments; treat Handler's set as a *re-transformed subset of
  our test split* (not an independent test set); exclude the 11 matching train items from any training data if the
  Handler set is used for evaluation. Cross-split near-duplicates are checked in Phase 2.

**D012 — Two-level bootstrap for every fine-tuned comparison (carried over from ecg-research D014)**
- Lesson from the previous project: comparing two runs of the *same* method with an item-only bootstrap on
  seed-averaged predictions gave a spurious p < 0.001; resampling training seeds and test items together gave
  p = 0.19 (correct). With 3 seeds this test is conservative, so borderline effects may need 5 seeds.
- Implemented in `src/evaluation/stats.py::two_level_bootstrap` (copied/adapted now so this repo no longer depends on
  ~/Desktop/ecg-research), with a unit test that identical methods come out non-significant.

**D013 — OpenBioLLM-8B kill-test A numbers are not used**
- Decision: treat OpenBioLLM-8B's letter-probability results as invalid; do not report them as measurements.
- Evidence: mean probability on A–D tokens is 0.11 (0.003 under few-shot) vs. 0.999 for Qwen/Llama
  (`results/pilot/killA/aaditya__Llama3-OpenBioLLM-8B/n300_s0/summary.json`, `mean_letter_mass`).
- Alternatives: re-run with generation + parsing, or a different medical model; decided after Checkpoint 1.
- Reasoning: argmax over letters that hold 11 % of the mass does not reflect the model's actual answer.

**D014 — Stage-2 kill test: decisive arm first, secondary arms conditional**
- Decision: Kaggle batch `killtest2` runs only the pre-registered decisive arm (`all7`) for Qwen and Llama plus the
  education-prompt side-cost job; `heldout4` and `sft_clean` (`killtest2b`) run only if `all7` passes.
- Alternatives: all three arms in one batch (~12 GPU-h).
- Reasoning: if `all7` fails, method A stops and the secondary arms answer nothing; saves ~6 GPU-h of the weekly quota.
- Also: training fp16 non-finite micro-batches are skipped and counted (reported in train_info); LoRA hyperparameters
  fixed a priori (no search) — this is a go/no-go test with 1 training seed, not a reported result.

**D015 — Checkpoint 1b: GO method A; method B dropped as a method**
- Decision (approved): continue with counterfactual-invariance fine-tuning (axis A) as the paper's method.
  Axis B (generate-then-verify) failed its pre-registered kill test on both models and becomes an analysis section.
- Framing: plain SFT and BCT are mandatory baselines; held-out hint families are the key generalisation claim.
- Next: Phase 2 (data).

**D016 — Phase 2 data decisions**
- New held-out families (ours): authority, external_anchor, automation, convenience; 2 templates each, plus education
  texts in the BiasMedQA style. `BIASMEDQA` is pinned by name, so `all7` cannot silently change.
- Training pool excludes train-01318 and train-03960 (duplicate option texts). Pair sampling now draws from this
  filtered pool, so post-kill-test pair sets differ from the kill tests (those stay tied to commit a460ff4).
- Templates frozen now. Item validity review → exclusion + validity rate only.
- Third model shortlist: Llama3-Med42-8B (primary), MedGemma-4B-it (backup); decided by a Kaggle letter-mass probe.
- Alternatives: paraphrase-only held-out sets (rejected, D007); editing templates after review (rejected: leakage risk).

**D017 — Phase 3 reproduction design**
- Decision: reproduce the BiasMedQA protocol faithfully (5-option MedQA, the paper's Methods prompt incl. "for research
  purposes only", options on one line, the paper's one-/few-shot headers) on the **dev** split, using a fixed random subset
  of 600 questions (seed 0) for every model. gpt-4.1-mini runs all 4 strategies (API, ~$3.3). Qwen and Llama run none,
  education and few-shot (one-shot skipped to save GPU quota). Med42 runs none and education (doubles as the third-model
  probe; it stops early if letter mass < 0.5).
- Alternatives: the paper's test split (rejected: the test set stays untouched until the frozen Phase 5 evaluation);
  all 1,272 dev questions (API ~$7, GPU ~2x).
- Reporting: accuracy with non-responses excluded (the paper's convention) and strict; paired bootstrap + McNemar, plus
  the paper's two-proportion z-test.
- Finding while reading the supplement: the paper's per-model and frequency/recency headline drops are percentage
  points, but its false-consensus (24.9) and confirmation (8.1) headlines do not match its own Table 1 (17.1 and 6.3 pp).
  A plausible cause is the alternative template variants in its code (not verified).
- Note: the gpt-4.1-mini run started from the working tree just before commit 3f06ae1, with identical code (its
  provenance shows git_dirty=true).

**D018 — Token-budget batching after the repro1 out-of-memory failure**
- What happened: in `repro1`, Qwen and Llama ran out of GPU memory on the few-shot prompts (16 prompts of up to ~1.5k
  tokens per batch; stage 1 had used 8). Their no-bias, cue and education conditions completed. Med42 completed.
- Decision: `HFModel.letter_probs` now batches by a padded-token budget (default 8,192) and halves a batch on
  out-of-memory. Verified identical to batch size 1 (max probability difference 8e-6). Notebook sets
  `expandable_segments`. The missing few-shot conditions are re-run in `repro1b` and merged with
  `scripts/merge_repro_runs.py`; the overlapping conditions serve as a determinism check.
- Alternatives: re-run everything (~1 GPU-h more); upload partial records to Kaggle (impractical).

**D019 — GPU batch-composition noise; fixed evaluation settings for Phase 5**
- Finding: re-running the same conditions on Kaggle with different batching (repro1 vs repro1b) changed 0.8–2.7 % of
  answers. Accuracy moved by ≤ 1.0 pp per condition (mean 0.24 Qwen / 0.46 Llama). Llama's flips are near-ties (median
  top-2 margin 0.04). Qwen has some larger flips (90th percentile 0.46): fp16 4-bit Qwen is numerically sensitive.
- Decision: every Phase 5 evaluation of every method uses identical inference settings (batch_size, max_batch_tokens,
  4-bit config, library versions recorded). This noise floor is reported alongside the results.
- Also: prompt baselines are format-sensitive (few-shot +15.4 pp for Qwen in the faithful format vs ~0 in the kill-test
  format) → Phase 5 tunes each prompt baseline's format on dev, like our method's hyperparameters.

**D020 — Phase 5 plan frozen (Checkpoint 3)**
- Decision: the pre-registered plan in docs/preregistration.md Part B (as of commit 335ce58) is approved and frozen: training
  cues = 7 BiasMedQA templates, held-out = 8 new templates; arms base / oracle warning / generic warning / few-shot /
  SFT-clean / BCT / CIFT (+ Llama ablations no-KL, no-helpful); fixed hyperparameters for all trained methods (no
  search); 3 seeds; MedQA test; hypotheses H1–H4 with two-level bootstrap. Amendments only as dated notes with
  reasons, never after seeing test results.

**D021 — P5-0 outcomes: prefix evaluation accepted; best prompt baselines selected (frozen rule)**
- Prefix-shared evaluation accepted (amended B7.1): agreement with the gap1 standard path 99.2 % (Qwen) / 98.7 % (Llama),
  max per-condition accuracy difference 0.5 / 0.8 pp; 2.1× / 3.1× faster. `results/main/prefix_eval_check.json`.
- Best prompt baseline by mean dev misleading accuracy over 15 cues: **Qwen → few-shot** (45.5 vs oracle 44.2, generic
  33.9); **Llama → generic warning** (59.9 vs oracle 58.8, few-shot 41.2). Few-shot therefore runs on test for Qwen only.
- Noted risk: Llama's generic warning is a strong baseline (unseen-cue dev accuracy 59.6 vs base 33.1).

**D022 — P5-1 failure: test split missing from the Kaggle bundle**
- What happened: the bundle still excluded MedQA test (a Phase 1–4 safeguard). All 9 P5-1 jobs failed at evaluation:
  the two CIFT jobs after ~100 min of training, twice each (retry retrained). 3.28 GPU-h lost on the second Kaggle account.
  No result was produced or seen.
- Fixes: (1) the bundle includes test.jsonl and the notebook copies it; (2) run_main.py loads every input it needs
  BEFORE loading the model or training (fail fast); (3) trained adapters are saved and reused on a retry. Verified
  locally (adapter reuse) and by simulating the notebook setup with the real bundle.
- Lesson: local tests passed because test.jsonl exists locally. The notebook simulation now checks every split file.

**D023 — Pre-registered H1 vs BCT failed; honest reframing (per frozen rule B6)**
- Result: on held-out cues CIFT − BCT = +0.9 [−1.4, 3.1] (Qwen), +1.0 [−1.5, 3.4] (Llama). Not significant → the
  pre-registered general H1 is not supported. H1 vs base, best prompt and SFT-clean pass on both models.
- Reframing (to be confirmed; final wording after Med42 and the ablations): "Consistency training
  (BCT or CIFT) is what removes the vulnerability, generalising to unseen hint families; anchoring the consistency
  objective to gold labels (CIFT) additionally keeps the accuracy gains of fine-tuning, preserves more of the benefit of
  correct hints and transfers better" — the last part is EXPLORATORY (not pre-registered) and must be labelled so.
- Consequences for the remaining plan: (1) BCT on Med42 moves from optional to recommended (BCT is now the key
  comparison). (2) The no-KL ablation (H4) is now central: if gold-label training on cued prompts without KL matches
  CIFT, the contribution is the gold anchor, not the KL term. No hypothesis is changed.

**D024 — H4a (KL) fails; the method's active ingredients are gold-label counterfactual pairs + helpful controls**
- Pre-registered H4a: CIFT vs no-KL on held-out cues +1.7 [−1.8, 4.9], not significant; no metric differs. H4b
  (helpful pairs → helpful-cue accuracy) +30.2 [27.0, 33.2], supported.
- Reframing (with D023): the contribution is (1) the evaluation showing misleading-only benchmarks reward a contrarian
  shortcut and need helpful-cue controls; (2) gold-label counterfactual augmentation WITH helpful-cue controls as the
  recommended training recipe (robust like BCT, accurate like SFT, keeps the value of correct hints); (3) the KL term is
  reported as tested and unnecessary. All framing honest per integrity rule 7; exploratory items labelled.
- Plan consequence: no change to hypotheses. Med42 (P5-4) remains to test generality; BCT on Med42 recommended (D023).

**D025 — Med42 best prompt baseline (frozen dev rule)**
- Mean dev misleading accuracy over 15 cues: oracle warning 62.0, generic warning 58.7, few-shot 51.1 → **oracle warning**.
  Few-shot therefore not run on test for Med42 (few-shot amendment, docs/preregistration.md §3).

**D026 — Phase 5 required experiments complete; final hypothesis verdicts**
- All required arms done for 3 models × 3 seeds on MedQA test (+ Llama ablations). Verdicts in results/tables/main_results.md (summary table):
  H1 supported vs base, best prompt and SFT-clean on 3/3 models; NOT vs BCT (0/3). H2 vs SFT 2/3. H3 vs base 3/3, vs
  SFT 2/3 (Med42 fails the −2 pp bound). H4a not supported (KL unnecessary); H4b supported.
- Paper framing follows D023/D024 (honest): consistency training on cued copies is what removes the vulnerability and
  generalises to unseen hint families; gold-label anchoring + helpful controls make it the best all-round recipe; the
  KL term is unnecessary; misleading-only benchmarks reward a contrarian shortcut. Exploratory items labelled.

**D027 — Paper format: Elsevier elsarticle**
- Decision: LaTeX with `elsarticle` (preprint, numbered citations), compiled with tectonic. Fits the shortlisted Elsevier
  journals (Computers in Biology and Medicine, Artificial Intelligence in Medicine, Journal of Biomedical Informatics);
  convertible to IEEEtran if a conference is chosen.
- Every number in the text comes from `paper/numbers.tex` and every table from `paper/tables/`, both generated by
  `scripts/make_tables.py` from results/ (no hand-typed numbers). References verified by `scripts/verify_refs.py`.

**D028 — Item validity review completed; how the reviewer is described**
- Reviewer: a medical graduate (MBBS) who is not yet licensed to practise (preparing for postgraduate entrance). Described in the paper exactly so, by role, never as "physician/doctor/clinician"; not
  named unless she consents. (Replaces the earlier plan of an MBBS student.)
- Results (results/review/review_summary.{json,md}, scripts/analyze_review.py): valid 80/80; key still best 76/80 (all
  60 cued items OK; 4 failures are NOTA items, NOTA clearly correct 16/20); cue plausibility mean 2.1/5 (BiasMedQA
  templates 1.6, new families 2.6). As pre-registered, flagged items excluded in a sensitivity analysis only: held-out-cue
  comparisons unchanged. Templates not edited.

**D029 — Release preparation: reproducibility fixes (no result changed)**
- Added `scripts/analyze_exploratory.py`, which writes `results/tables/exploratory_cift_vs_bct.json` and
  `ablations_llama.json` (previously produced outside the release). It reproduces every shipped value exactly and adds the
  residual held-out drop for Med42, which the earlier file lacked.
- Provenance: `git_dirty` is now `None` (unknown) without a git checkout; before, every Kaggle run recorded `true`. The
  versions of transformers, PEFT, accelerate, bitsandbytes and datasets are now recorded; the Kaggle notebook installs
  pinned versions from `configs/requirements-kaggle.txt`. The shipped runs predate both changes.
- Answer parser: the letter in "answer is X" is case-sensitive, so "the answer is a …" is no longer read as option A.
  Re-parsing every shipped free-text reply with the fix changes no prediction.
- Reviewer path: `configs/queues/quick_check.txt` (one Kaggle session), `ingest_kaggle.py --dest`, and
  `scripts/compare_rerun.py` to compare a re-run with the shipped results.
- Wording: "item validity review" replaces "clinician review" in public files (D028); `make_clinician_sheet.py` renamed
  `make_review_sheet.py`. The pre-registration text is left unchanged.
