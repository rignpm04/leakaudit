# Pre-registration: leakaudit fine-tune run 5

**A small open model that audits tables for target leakage, tested blind on sealed real test 5**

- Author: Patrick Rigney
- Registered: 2026-10-27, as release v1.0-prereg of github.com/rignpm04/leakaudit (Zenodo concept DOI
  10.5281/zenodo.23680062). This document, the code it names and the fingerprints in section 10 are fixed from then on.
- Earlier, exploratory work (runs 1-4): release v0.1, DOI 10.5281/zenodo.23680063.

## 1. Question

Target leakage is a column whose value would not be known at the moment a prediction is made, or that is computed from
the outcome, or that identifies rows so a model can memorize outcomes. A leak makes a model look better in testing
than it can be in use. We ask whether a 9-billion-parameter open model (Qwen3.5 9B), fine-tuned on frontier-model
ratings of public tables and on generated tables with a known answer key, can find leaks in new real tables from the
table's description alone (column names, types, missing shares, example values, and three sentences on the task),
without the outcome values:

1. better than the same model untrained,
2. nearly as well as two frontier models,
3. without relying on recognizing the dataset (it is also tested on disguised copies), and
4. far better than the column check that commercial tools use.

## 2. Test set: real test 5 (sealed)

- 68 public tables (14 without leaks, 54 with leaks; 388 leak columns and 106 debatable columns in the sealed key).
  At least half of each table's rows were first published on or after 2026-07-01. No test 5 table, nor any table from
  the same file or with nearly the same columns, is used in training (checked by t3_make_training.py).
- The key was sealed on 2026-10-26 in test5_sealed_2026-10-26.zip, SHA-256
  `853b13b2724edf2aa26d2f0cbf7ddbea09b4120c7b434dc620916e0694337588`, published in release v0.1 (dated 2026-10-27
  UTC). The zip holds the key with its evidence, the review notes and the process files.
- Test 5 itself (the tables, their prompts, every answer) stays private until the results are published; only
  fingerprints are public before that. Third-party tables are never re-hosted: the results release gives their links
  and checksums.
- The 42 generated test tables of run 4 are used again: 24 from the 4 domains the generated training tables come from,
  12 from 2 held-out domains, and 6 trap tables (legitimate columns built to look late).

Key corrections (rule fixed at sealing): a label of the key changes only if (1) a human labeler of the 15 labeled
tables (section 4.5) disagrees with it, (2) an independent re-check finds documentation or an exact data relation
that supports the labeler, and (3) the change is made before any answer of the run 5 models on test 5 is scored or
opened. Every change is listed with its evidence (key_corrections.json). Results are reported on the corrected key
(primary) and on the sealed key.

## 3. Prompts

Each table is prepared as in runs 3-4 (at most 50,000 rows, the outcome renamed `target`) and described in prompts of
at most 10 rated columns: the unit of observation, the target, the prediction moment, the names of all columns, and
for the rated columns one line each with type, missing share, distinct values and examples (5 examples, seed 7).
Thinking is off for the small model. Every model answers the same prompts with a short reason, a category and a risk
of 0-100 per column.

- **Real test 5:** 208 prompts for the 68 tables.
- **Real test 5 disguised:** 179 prompts for the 65 tables that have a usable disguise. GPT wrote,
  once per table, new column names, labels and task sentences that do not give the dataset away; numbers are shown
  only as rounded ranges (2 significant figures) or levels, dates are shifted by 7-15 years, and the columns are
  shuffled. The answers are mapped back to the real columns.
- **Generated test tables:** 127 prompts.
- All 514 prompts are in test_prompts.jsonl (fingerprint in section 10). The file holds no answer key.

## 4. Methods

### 4.1 Models

| label | model | answers per prompt |
|---|---|---|
| untrained | unsloth/Qwen3.5-9B, 16-bit, thinking off | 3 samples (temperature 0.6, top-p 0.95, top-k 20); a column's risk is their mean |
| trained s3407, s3408, s3409 | the same with a LoRA add-on, three training runs identical but for the seed | 3 samples, as above |
| GGUF Q4_K_M, Q6_K | the trained run with the lowest validation loss, merged and quantized for laptops; llama.cpp | 1 greedy answer |
| Claude | claude-sonnet-5-5 (Anthropic API) | 1 answer |
| GPT | gpt-6.1-sol (OpenAI API) | 1 answer |

A prompt whose answer leaves a column unrated (or is unreadable) is asked once more with a reminder (run 3's rule).
The small models answer 16 prompts at a time (left padding); they answer one at a time only if a check on 16 prompts
finds 2 or more answers readable in one mode and not the other.

### 4.2 Training (unchanged from run 4 except where noted)

- Examples: one chunk prompt + its answer. Public training tables (268 + 433): the average of Claude's and GPT's
  ratings, rounded to 5; each table twice, as is and **disguised** (new in run 5; reasons that would name the real
  table are replaced by a short reason). Development tables (27 + 47 + 55 + **real test 4's 56**, new in run 5): the
  teachers' average kept on the side of the documented key (a documented leak at least 80, a debatable column 20-80,
  any other column at most 20); three times each: as is, disguised, and with shuffled columns and 60-95% of the rows.
  800 generated tables from 4 domains: the generator's key sets the risk (95 or 5), GPT's reason when it agrees.
- train.jsonl and val.jsonl (5% of the tables, whole tables) as fingerprinted in section 10.
- LoRA rank 32, alpha 32, on the attention and MLP projections; 2 epochs; learning rate 1e-4, cosine schedule, 3%
  warm-up; batch 4 x 4 gradient accumulation; examples up to 8,192 tokens; the loss counts only the answer; the
  validation loss every 100 updates; the checkpoint with the lowest validation loss is kept. Seeds 3407, 3408, 3409.

### 4.3 Baselines without a language model

- **Column check** (the approach of DataRobot's and H2O's leakage warnings): each column alone predicts the outcome,
  5-fold out of fold, with gradient-boosted trees (scikit-learn HistGradientBoosting, 60 rounds, at most 8,000 rows).
  Score 0-100 = 100 x |2 AUC - 1| for a yes/no outcome, 100 x R^2 (at least 0) for a number. It uses the outcome
  values, which the language models never see.
- **Keyword rule:** a column is flagged when its name holds one of 40 fixed words (date, status, total, paid, ...;
  r5_common.py).

### 4.4 Frontier answers, recognition and the sentences-only check

Claude and GPT answered every test 5 prompt (real and disguised) before this registration, blind (the folder holds no
key). They were also asked, per table and condition, which public dataset the table comes from (recognition), and, from
the three task sentences alone, how likely a leak is (sentences-only check). All these answers are fingerprinted in
frontier_manifest.json (section 10). A guess recognizes a table if it names its source or its edition, by the rules of
test5_aliases.json, written before any guess was read (fingerprint in section 10).

### 4.5 Human labels

15 of the 68 tables (4 without leaks, 11 with; 257 columns, chosen by a fixed seed with limits per family and topic)
were put in a workbook that shows the lines the models see: identical in 240 of the 257 rows; in the other 17, the
example lists show equally frequent categories in another order or choice (the workbook was made with another pandas
version, which breaks ties differently). Labelers mark each column leak or fine, alone,
without searching for the tables. Labels returned by 2026-12-15 are used. A labeler who is an author of this study is
named as such in the results, with whether the labels were made before the author could see the key.

## 5. Outcomes

- A model's ratings of a table are its answers to the table's prompts. A prompt with no readable answer flags nothing.
  Debatable columns count neither as found nor as false flags.
- **Macro recall** of a group: the mean, over its tables with leaks, of the share of each table's leak columns
  flagged.
- **Within budget (primary):** each model at its best threshold (0-100 in steps of 5) whose false flags total at most
  1 per table of the group; ties go to the higher macro recall, then to fewer false flags.
- **Fixed (secondary):** a column is flagged at risk >= 50.
- **Trained** = the mean of the three seeds' scores.

## 6. Hypotheses (all within budget, on the primary key)

| | comparison | supported if |
|---|---|---|
| H1 | trained - untrained, real test 5 | >= +0.10, and the 95% interval above 0 |
| H2 | best frontier model (Claude or GPT) - trained, real test 5 | <= +0.05 |
| H3 | trained on real test 5 - trained on its disguised copies (the same tables) | <= +0.05 |
| H4 | trained - column check, real test 5 | >= +0.20, and the 95% interval above 0 |
| H5 | trained - untrained, held-out generated domains | >= -0.05 |

Intervals: 2,000 bootstrap resamples of the tables of each group (paired: every model on the same resampled tables;
H3 resamples the tables with a disguise, both conditions together), seed 20261101, the threshold tuned again in
every resample; the 2.5th and 97.5th percentiles. All five are reported whatever they show; there is no correction
for multiple comparisons, and no hypothesis is added or dropped after the results.

## 7. Secondary analyses (reported, no verdict)

Fixed-threshold results; each seed and the spread between seeds; the GGUF files; Claude and GPT on each group and
their own disguise drop; the column check at the vendors' default levels (DataRobot Gini >= 0.85 and >= 0.975; H2O
AUC >= 0.95 / R^2 >= 0.95 and the 0.80 notice level) and the leak columns below H2O's notice level that the trained
runs flag; the keyword rule; the human labelers and every model on the 15 labeled tables (both keys), and the
agreement between labelers; recognition of the source and of the edition, real and disguised; the sentences-only
check (AUC of the stated likelihood, tables with leaks against tables without); the trap tables at the fixed
threshold (leaks found, trap columns flagged). Anything else is labeled exploratory.

## 8. Order of events

1. Done before registration: test 5 sealed and its fingerprint published (v0.1); the training files, the test
   prompts, the disguises, the frontier answers and the baselines made and fingerprinted on the author's computer,
   without the key.
2. This registration is published (Zenodo v1.0-prereg).
3. One RunPod session trains and runs the small models on the test prompts, without the key (it never goes to the
   pod). The fingerprints of every answer file (answers_manifest_run5.json) are published at once, before step 5.
4. The hand labels come back.
5. The key is unsealed: t7_label_review.py lists the labelers' disagreements; each is re-checked independently;
   key_corrections.json is written (possibly empty).
6. t8_score.py scores everything as fixed here; the results are published with test 5, its key and the answers
   (except the frontier models' raw answers if their terms do not allow it), as release v1.0.

## 9. Deviations

Any departure from this plan (a failed step, a changed setting, an answer that could not be obtained) is reported with
the results, with its reason and its effect where it can be measured. The only edit to the registered code that is
planned is entering the published fingerprint of the pod's answer files in t8_score.py (ANSWERS_FILES_SHA256: the
SHA-256 of answers_manifest_run5.json's sorted list of answer files and their SHA-256).

## 10. Fingerprints (SHA-256)

| file | what | SHA-256 |
|---|---|---|
| test5_sealed_2026-10-26.zip | the sealed key | `853b13b2724edf2aa26d2f0cbf7ddbea09b4120c7b434dc620916e0694337588` |
| train.jsonl | training examples | `4ea8fd0233b0303099ab6fbacdfc490488f5c29dcddb8f8f9ceae3fdb572c0e3` |
| val.jsonl | validation examples | `c322c45c13332e2196673a11d1671992ec493a13aa035d3cf7f6d9d20858394f` |
| test_prompts.jsonl | every test prompt (private until the results) | `18d6a8e97b046c0a87398a950360bd59b95295e038b1df5ad5f2feb7e56c3e59` |
| frontier_manifest.json | every Claude and GPT answer file on test 5: the SHA-256 of its sorted list of "path sha256" lines | `7ad2ac42ea14a068670ec7c7312356358c3cf10625dfc3561dbb09779f113fe5` |
| baselines_test5.jsonl | the column check and keyword rule on test 5 | `f166ff85b3ee7fcc76512696a73c2e206a5975e36780c560c7416aa05220bef4` |
| test5_aliases.json | what counts as recognizing a table (private until the results) | `ca81dbb53ce50bd7d17055173a5d872d78f9979ceccee5b1d88906da8ae10615` |
| leak_labeling_test5_15_tables.xlsx | the blank labeling workbook | `e2be28061c6b93a3ad14a2236f43dba625facd97e31818a747103c0364b82fe6` |

The code of this release (every script and the notebook) is in CODE_SHA256.txt.
