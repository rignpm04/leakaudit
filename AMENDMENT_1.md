# Amendment 1 to the run 5 pre-registration

Written on 2026-10-28, after the RunPod session and before test 5's answer key was opened or any run 5 answer
was scored. It records one deviation in the training run and when scoring starts. Sections 1-7 of
PREREGISTRATION.md (question, test set, prompts, methods, outcomes, hypotheses, analyses) do not change.

## 1. The training run (section 8, step 3)

One RunPod session on an A100-SXM4-80GB, 2026-10-28 02:07-08:30 UTC: torch 2.8.0+cu128, transformers 5.2.0,
trl 0.22.2, Unsloth and unsloth_zoo 2026.10.9 (the notebook installs the newest Unsloth; run 4 had 2026.10.7),
Python 3.12.3. Test 5's key never went to the pod.

**Deviation 1.** Seed 3408 stopped at its first training step with an error inside Unsloth (TypeError:
`_legacy_causal_conv1d_fn() missing 1 required positional argument: 'hidden_states'`). Unsloth 2026.10.9's
per-block compile wraps the GatedDeltaNet short convolution in a function whose first argument is named
`hidden_states`, while transformers 5.2.0 passes it as `x`. One line was added to the notebook's settings cell,

    os.environ["UNSLOTH_BLOCK_COMPILE"] = "0"

which turns the per-block compile off. The kernel was restarted and the notebook run again; the finished steps
were kept.

- Made before the fix (Unsloth's default, per-block compile installed): the untrained answers, and seed 3407's
  training and answers. Made after it (per-block compile off): seeds 3408 and 3409, the GGUF files and their answers.
- Unsloth uses the per-block compile only in training: it skips it in evaluation mode and whenever a generation
  cache is in use. Every model therefore answered through the same code. In training it compiles the same
  computation, which can change rounding, speed and memory, not the settings.
- Validation loss of the kept checkpoints (step 700 of 712 for each): seed 3407 0.2241, seed 3408 0.2241, seed 3409 0.2243.
- The notebook as run differs from the registered train5_runpod_run5.ipynb by that one line only
  (train5_runpod_run5_as_run.ipynb here, outputs removed). SHA-256:
  - registered: `e6c5bb456ad573d447972407991e0ee8b44fdf5c4ac3fcaa2f5363c4a1379342`
  - as run, here: `54dd810af49aaec89296c0c02bf4a4cbaf83c16b85373b9f2909addbcd9e8df9`

**The laptop files.** Seeds 3407 and 3408 tie at the recorded validation loss; the registered code breaks ties by the
lower seed, so the GGUF files come from seed 3407. They were converted with Unsloth's prebuilt llama.cpp (b11876) and
answered with llama-server built on the pod (llama.cpp commit f4d8af6, CUDA). SHA-256:
  - Q4_K_M (5.38 GB): `8ddf4acf0b384fe75821e18a0b6db4743c4a4fbfacc292cfac7a59c3880dae73`
  - Q6_K (7.04 GB): `1839bfcf85d09704ee5ed58076f479dae2ca95e9be14c9920c79670f289bc5c8`

**Answers.** Each of the six models gave a readable answer to all 514 prompts. Between 1 and 3 prompts per model
were asked again under the registered retry rule; between 0 and 3 columns per model stayed unrated, and they count
as not flagged.

## 2. Published here, before the key is opened

| file | what | SHA-256 |
|---|---|---|
| answers_manifest_run5.json | SHA-256 of the six answer files (untrained; seeds 3407, 3408, 3409; GGUF Q4_K_M and Q6_K), of the three input files (the same as section 10 of the pre-registration), of the run record and of the three seed records | `c7521fb6ca26b0e944784823d953227f51a44413666bc0edf5df766daa5f614d` |
| train_run5.json | the run record: settings, versions, losses, timings, answer counts (no test content) | `962eb366361a75e6bfaf1dac722f15df78d0b937a47aa41e3a75276eacfab093` |
| adapters_sha256_run5.txt | SHA-256 of every file of the three trained LoRA add-ons | `ae3ff7835d3584d3cfea4de8ff3bf313db63cb072123e41c2a06000efe51181e` |
| train5_runpod_run5_as_run.ipynb | the notebook as run, code only | `54dd810af49aaec89296c0c02bf4a4cbaf83c16b85373b9f2909addbcd9e8df9` |
| t8_score.py | the scoring script with the answer files' fingerprint entered (section 9's one planned edit); registered version `89f9954b7e56a3aa268ce2524db5448a43c198d611357746a34283abb9678edd` | `bb4a83682e89355beb02f12fbb327adabf22d518830cae212d612ffc5ada74a8` |

The answer files' fingerprint, which t8_score.py checks (SHA-256 of the sorted "file sha256" lines of
answers_manifest_run5.json): `ff1c61ceb847d2bbac859425abad2c57118dad36c76a95628ec317e2919b0cb2`.

The answers themselves and the trained add-ons are not part of this release; their fingerprints are.

## 3. When scoring starts (section 8, steps 4-6)

Scoring may start before the 2026-12-15 deadline for labels, as soon as the author's hand labels are in; the author
labels before opening the key and is named as an author-labeler in the results (section 4.5). As the sealing rule
already requires, key corrections can only come from labels returned before the run 5 answers are scored. Labels
returned after scoring has started are used only for the comparison of labelers and models and for the agreement
between labelers (section 7); labels returned after the results are published, which reveals the key, are not used.
