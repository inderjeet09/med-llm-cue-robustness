# Prompt templates

Every template used by an experiment is stored here as a versioned file and never edited after use. Changing one
means adding a `_v2` file. All templates are **frozen before any test-set evaluation**.
Each result file records the template names it used. Placeholders use `{name}` and are filled by `src/llm/prompts.py`.

| file | source | used for |
|---|---|---|
| biasmedqa_instruction_v1.txt | Schmidgall et al. 2024, npj Digit Med (CC BY 4.0), quoted | letter-only MCQ prompt (axis A) |
| cot_mcq_v1.txt | ours; zero-shot chain of thought as in Bedi et al. 2025 | axis B baseline with a visible NOTA option |
| handler_baseline_v1.txt / handler_safety_v1.txt | Handler, Bedi & Shah 2026, github.com/handler9/NOTA-Benchmark @c8db360 (MIT); the hard-coded "A, B, C, or D" became `{letters}` | axis B prompt baselines |
| gtv_step1_v1.txt / gtv_step2_v1.txt | ours (method B) | option-blind generate-then-verify |
| biasmedqa_faithful_instruction_v1.txt | Schmidgall et al. 2024 Methods (CC BY 4.0), quoted verbatim incl. "for research purposes only" | faithful BiasMedQA reproduction (Phase 3) |
| biasmedqa_demo_incorrect_v1.txt / biasmedqa_demo_correct_v1.txt / biasmedqa_next_question_v1.txt | Schmidgall et al. 2024 Methods, one-/few-shot example headers, quoted | faithful reproduction |
| generic_warning_v1.txt | ours (Phase 4, frozen at Checkpoint 3) | realistic prompt baseline: one warning covering all cue types, appended to the instruction (unlike the paper's per-bias education text, it does not know which cue is present) |
