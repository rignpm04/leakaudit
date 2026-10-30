# Run 5: results

Run 5 was pre-registered in release v1.0-prereg ([10.5281/zenodo.23685637](https://doi.org/10.5281/zenodo.23685637)).
Its answers were fingerprinted and published in v1.0-answers
([10.5281/zenodo.23725257](https://doi.org/10.5281/zenodo.23725257)) before test 5's key was opened. It was scored on
2026-10-30 with the registered `t8_score.py` (SHA-256 `bb4a83682e89355beb02f12fbb327adabf22d518830cae212d612ffc5ada74a8`, published in v1.0-answers) against the sealed key
(SHA-256 `853b13b2724edf2aa26d2f0cbf7ddbea09b4120c7b434dc620916e0694337588`, published in v0.1). The scoring checked the answer files against their published fingerprint
(`ff1c61ceb847d2bbac859425abad2c57118dad36c76a95628ec317e2919b0cb2`).

**No key correction.** The author's hand labels disagreed with the sealed key on 12 columns. An independent re-check
of each found no documentation or exact data relation supporting a change, so the primary key is the sealed key
(`key_corrections.json`). Deviations from the plan are listed in `AMENDMENT_1.md` (v1.0-answers).

## The five pre-registered hypotheses

Within budget: each model at its best threshold with at most one false flag per table on average; macro recall = the
mean, over the tables with leaks, of the share of each table's leak columns flagged. Trained = the mean of the three
training seeds. Intervals: 2,000 paired bootstrap resamples of the tables, threshold tuned again in each.

| | comparison | estimate | 95% interval | rule | verdict |
|---|---|---|---|---|---|
| H1 | trained − untrained, real test 5 (68 tables) | +0.236 | +0.111 to +0.336 | ≥ +0.10 and interval above 0 | supported |
| H2 | best frontier model (GPT) − trained, real test 5 | +0.108 | +0.049 to +0.203 | ≤ +0.05 | not supported |
| H3 | trained on real − trained on disguised copies (the same 65 tables) | +0.088 | −0.006 to +0.186 | ≤ +0.05 | not supported |
| H4 | trained − column check, real test 5 | +0.475 | +0.299 to +0.599 | ≥ +0.20 and interval above 0 | supported |
| H5 | trained − untrained, held-out generated domains (12 tables) | +0.164 | +0.007 to +0.253 | ≥ −0.05 | supported |

Three of the five held. Fine-tuning improved the model on the sealed real tables (H1) and on generated domains held
out of training (H5), and the trained model beat the vendor-style column check by a wide margin (H4). It did not come
within 0.05 of the best frontier model (H2), and it lost more than 0.05 on disguised copies of the tables (H3).

## Macro recall within budget

| model | real test 5 (68) | real, the 65 with a disguise | disguised copies (65) | held-out domains (12) | training domains (24) |
|---|---|---|---|---|---|
| Untrained Qwen3.5 9B | 0.599 | 0.610 | 0.518 | 0.525 | 0.536 |
| Trained, seed 3407 | 0.826 | 0.807 | 0.719 | 0.742 | 0.865 |
| Trained, seed 3408 | 0.847 | 0.845 | 0.782 | 0.667 | 0.873 |
| Trained, seed 3409 | 0.831 | 0.820 | 0.704 | 0.658 | 0.882 |
| GGUF Q6_K (seed 3407) | 0.804 | 0.801 | 0.680 | 0.567 | 0.861 |
| GGUF Q4_K_M (seed 3407) | 0.736 | 0.733 | 0.658 | 0.642 | 0.836 |
| Claude (claude-sonnet-5-5) | 0.909 | 0.902 | 0.829 | - | - |
| GPT (gpt-6.1-sol) | 0.943 | 0.942 | 0.883 | - | - |
| Column check | 0.360 | 0.364 | - | - | - |
| Keyword rule | 0.000 | 0.000 | - | - | - |
| **Trained, mean of the three seeds** | **0.835** | **0.824** | **0.735** | **0.689** | **0.873** |

## Other pre-registered analyses

`results_run5.txt` holds every analysis fixed in the pre-registration: the same results at a fixed threshold (risk
50), each seed, the trap tables, the column check at the vendors' default levels (314 of the 388 leak columns
lie below H2O's notice level), the keyword rule, the hand labels against every model on the 15 labeled tables,
recognition of the tables by the frontier models, and the sentences-only check. `results_run5.json` holds the same
numbers in full precision.

## Not yet public

Test 5 itself (the tables' sources and checksums, the key with its evidence, the prompts), every answer, and the hand
labels with their review stay private until the outside labeler has returned the workbook (by 2026-12-15 at the
latest), because they reveal the key. They will be published in release v1.0 together with the paper. The frontier
models' raw answers will be included only if their providers' terms allow it.
