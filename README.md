# leakaudit

Target leakage is a column whose value would not be known at the moment a prediction is made, or that is computed
from the outcome. It makes a model look better in testing than it can be in use. leakaudit asks whether a small open
model (Qwen3.5 9B), fine-tuned on frontier-model ratings of public tables and on generated tables with a known key, can
find leaking columns from a table's description alone, without the outcome values, and how it compares with frontier
models, with the statistical column check that AutoML tools use, and with a human labeler.

## Releases

All releases are versions of one Zenodo record (concept DOI
[10.5281/zenodo.23680062](https://doi.org/10.5281/zenodo.23680062)).

| release | what | DOI |
|---|---|---|
| v0.1 | Exploratory runs 1-4, and the fingerprint of the sealed answer key of test 5 | [10.5281/zenodo.23680063](https://doi.org/10.5281/zenodo.23680063) |
| v1.0-prereg | Pre-registration of run 5: plan, hypotheses, code and fingerprints, before training | [10.5281/zenodo.23685637](https://doi.org/10.5281/zenodo.23685637) |
| v1.0-answers | Fingerprints of every run 5 answer and of the trained add-ons, before the key was opened; Amendment 1 | [10.5281/zenodo.23725257](https://doi.org/10.5281/zenodo.23725257) |
| v1.0-results | Run 5's results (`RESULTS_run5.md`) | this release |
| v1.0 (planned) | Test 5 opened, every answer, the hand labels, and the paper | |

## Run 5 in one paragraph

On 68 sealed real tables, with at most one false flag per table, fine-tuning raised the 9B model's macro recall from
0.60 to 0.83 (mean of three training seeds). The vendor-style column check reached 0.36; Claude and GPT reached 0.91 and
0.94. Three of the five pre-registered hypotheses held; details in `RESULTS_run5.md`.

## Checking the files

Every fingerprint is a SHA-256 of the file's bytes (`shasum -a 256 <file>` on a Mac). Sets of files are fingerprinted
as the SHA-256 of their sorted "path sha256" lines (`r5_common.canonical_sha` in the run 5 code).
