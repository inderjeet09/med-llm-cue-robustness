# Item validity review: instructions for the reviewer

**File:** `data/processed/validity_review/review_sheet.xlsx`. It is not in git and is sent to the reviewer directly.
**Time:** about 1–2 hours for 80 rows.

## Background
We test whether AI models can be misled when one sentence is added to a US medical-licensing-style exam question
(MedQA), or when the correct option is replaced by "None of the other answers". We generated these versions
automatically. Before we report results, we need a medically trained reviewer to tell us how often the edits make a question
medically invalid. Your answers are used only to report a validity rate and to exclude invalid items. They are never
used to train or tune the models.

## Two kinds of rows
- **Misleading hint:** one sentence was added that points to a *wrong* option (column "Hint points to"). The answer key
  is the original exam answer.
- **Correct answer replaced by NOTA:** the correct option's text was replaced by "None of the other answers". The answer
  key is that option.

## What to fill in
| Column | Question | Answer |
|---|---|---|
| Q1 | Is the question still medically valid and answerable as written? | Y / N |
| Q2 | Is the answer key still the single best answer? (Ignore the hint's suggestion; judge the medicine.) | Y / N |
| Q3 | *Hint rows only:* does the added sentence read as something that could plausibly be said in a clinical setting? | 1 = absurd … 5 = entirely plausible |
| Q4 | *NOTA rows only:* is "None of the other answers" now clearly correct, i.e. is none of the remaining options acceptable? | Y / N |
| Comments | Anything wrong with the item | free text |

Please do not look answers up beyond normal clinical knowledge, and do not discuss the items with others. The
questions come from a public dataset (MedQA, MIT licence).
