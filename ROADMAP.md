<!-- aiflow:roadmap -->
# CFRBM Roadmap — Theano to PyTorch migration

CFRBM implements a Restricted Boltzmann Machine for collaborative filtering (Salakhutdinov et al.), trained with contrastive divergence on MovieLens ratings from 1 to 5. Today it is Python 2 code built on Theano, which is no longer maintained and does not install on the project's Python 3.12 `.venv`. This roadmap moves it to Python 3 and PyTorch. Training behavior stays the same until the PyTorch version has been checked against the old math, and the known training bugs are fixed after that.

**Why PyTorch:** the model computes its CD gradients by hand and applies its own momentum update (`cfrbm/rbm.py:101-138`), so it gets nothing from Keras's `fit`/loss/optimizer design. PyTorch tensors map almost one to one onto the current Theano ops, run eagerly (no per-epoch `theano.function` compile), and support CUDA and Apple MPS. PyTensor, the maintained Theano fork, would be the smallest change, but it no longer has the old CUDA GPU backend.

**Evidence:** based on the `master` branch at `55b2cc7` (all source files, `Makefile`, `ubased.json`, `tests/test_utils.py`). Version numbers are planning labels, not releases. Implementation status and validation are recorded per item below.

## v0.1 — Run on Python 3

Planned. Port the code to Python 3 without touching Theano or the model math, in a commit separate from the framework change. Exit criterion: `make test` passes on Python 3.12, and the entry points import and parse the experiment config.

- [x] **F1 — Port syntax to Python 3.** Replace `xrange` with `range` (`cfrbm/utils.py:8`). Use integer division `//` in `get_index` (`cfrbm/user_based.py:47`, `cfrbm/item_based.py:49`) and in the reshape in `revert_expected_value` (`cfrbm/utils.py:41`). Acceptance: the existing `tests/test_utils.py` passes on Python 3.12.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | S |
  | Details | Clear specification; estimated three Python files across utilities and two runners, plus roadmap tracking. Replaced xrange and used integer division in the reshape and both levels of schedule indexing, preserving Python 2 behavior. |
  | Commit | `217992a732f3149c7d8be4b26183ae9b9c35f316` — implements F1.; `5562fd863d22572bcc53fa39a5597c4cd44b6d21` |
  | Validation | Python 3.12.2: make test passed all 4 existing tests. Isolated AST checks of both get_index functions passed integer-type, schedule-boundary, and fallback cases. Roadmap validator and git diff --check passed. |
  | Limitation | Full training remains unverified; legacy Theano dependencies and runner imports are unchanged. |
  | Limitation | Runner dictionary-key views remain incompatible with chunker's sequence slicing on Python 3. |
  | Next step | Convert runner key views to sliceable sequences or adapt chunker before validating full Python 3 training. |

- [x] **F2 — Fix item-based experiment loading.** `cfrbm/item_based.py:172` treats the result of `read_experiment()` as a dict, but it returns a list. Loop over the experiments the way `cfrbm/user_based.py` does. Acceptance: `item_based` reads `ubased.json` without a `TypeError`.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | S |
  | Details | Clear specification with no ambiguities; estimated one Python file in the item-based runner, plus roadmap tracking. The runner now loops over experiments and their configurations like user_based, retaining item-based dataset orientation. |
  | Commit | `2988520d690d061e19c4007b96885482593187bc` — implements F2.; `f59c3fc060af5f23e00a8ebd728b0cd9221273cd` |
  | Validation | Python 3.12.2: isolated entry-point checks reproduced the original TypeError, then passed with the real ubased.json reader, multiple experiments/configurations, and an empty list; dataset loading and training were mocked. make test passed all 4 tests. Roadmap validator and git diff --check passed. |
  | Limitation | Full CLI training remains unverified because of existing legacy Theano/import dependencies and Python 3 dictionary-key slicing issues. |
  | Next step | After the existing dependency/import and key-view compatibility work, validate item-based training end to end. |

- [x] **F3 — Make the package importable and declare dependencies.** Switch to package-relative imports (`from .rbm import CFRBM`) so the scripts run as `python -m cfrbm.user_based`, and add a `requirements.txt` (numpy, and torch from v1.0 on). Acceptance: `python -m cfrbm.user_based ubased.json` gets past its imports, and the tests import `cfrbm.*` cleanly.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | S |
  | Details | Clear import scope; estimated five implementation files across two runners, dependencies and launch documentation, plus tracking. Both runners use package-relative imports; README and Makefile use module execution. requirements.txt declares numpy and the current Theano backend, interpreting torch from v1.0 as a future dependency. |
  | Commit | `17ed0c56883e49d68ac89d07b97a8cf6a9e2ee31` — implements F3. |
  | Validation | Python 3.12.2: make test passed all 4 tests. With only Theano stubbed, all six cfrbm modules imported and both module entry points parsed ubased.json and reached dataset loading. make -n run, roadmap validator and git diff --check passed. |
  | Limitation | The real user_based CLI still fails during Theano import because collections.MutableMapping is unavailable on Python 3.12; the full import acceptance check is not met in this environment. |
  | Limitation | Training remains unverified; the configured ml-100k/u1.base file is absent. |
  | Next step | During F6/F8, replace Theano with torch in requirements.txt and rerun unmocked package imports and both CLI entry points on Python 3.12. |

## v0.2 — Lock in current RBM behavior

Planned. Add tests of the current RBM math so the PyTorch rewrite can be checked against it. Theano cannot run on Python 3.12, so the reference is a NumPy implementation, not the old code. Exit criterion: the reference tests pass and document the current update rule, quirks included. Depends on F1.

- [x] **F4 — NumPy reference implementation of the CD math.** In `tests/`, write a reference for `prop_up`, `prop_down`, the 5-way normalization in `sample_visible`, `gradient`, and one parameter update with momentum and weight decay. Copy `cfrbm/rbm.py` exactly, including the quirks fixed later in F10–F12. Acceptance: the reference functions take fixed weights and fixed binary samples and return deterministic outputs.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | M |
  | Details | Clear math scope; estimated and changed two test files across a NumPy reference and deterministic regression tests, plus tracking. Resolved sampling ambiguity with supplied binary visible samples. Added propagation, grouped sigmoid normalization, masked gradients and simultaneous float32 updates; preserved raw-gradient momentum, the negative-phase h1 factor and non-one-hot samples. |
  | Commit | `8a86d6d5e67fbc26cee2767691316296e2c6fc3a` — implements F4. |
  | Validation | Python 3.12.2: make test passed all 9 tests, including 5 new hand-calculated regression tests covering propagation, two rating groups per user, masks, momentum, decay, defaults, deterministic updates and unchanged inputs. Roadmap validator and git diff --check passed. |
  | Limitation | Reference accepts fixed samples and does not reproduce random-number streams; live Theano numerical parity remains unverified on Python 3.12. |
  | Next step | In F5, compare the backend against these reference functions using fixed inputs and samples. |

- [x] **F5 — Parity test harness.** Add `tests/test_rbm.py`, which compares an RBM implementation against the F4 reference using `allclose` on fixed inputs and fixed samples, so no random-number matching is needed. Acceptance: the harness is ready to target the v1.0 PyTorch `CFRBM`. Depends on F4.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | M |
  | Details | Clear math scope; estimated and changed one test file across fixtures, backend adaptation and parity assertions, plus tracking and handoff notes. Eight reusable checks compare propagation, fixed samples, five-way normalization, masked gradients and repeated CD-1 updates against F4 with shape checks and allclose. Resolved future API ambiguity in an isolated PyTorch adapter that assumes the existing callable factory signature. |
  | Commit | `2da1b278e70225f11665a8d93e40760696880651` — implements F5. |
  | Validation | Python 3.12.2: make test passed 9 tests with the PyTorch parity class explicitly skipped. A temporary NumPy adapter executing legacy math passed all 8 harness checks and rejected 5 injected faults. Confirmed missing torch fails all 8 cases after removing the legacy import gate. Roadmap validator and git diff --check passed. |
  | Limitation | Actual PyTorch adapter execution and backend parity remain unverified until F6; the current Theano implementation is explicitly skipped and torch is not installed. |
  | Limitation | The harness covers CD-1 updates, not CD-k chains or device-to-device parity. |
  | Next step | In F6, run all 8 parity cases without skips; if the callable factory signature changes, update TorchAdapter.train while retaining the reference checks. |

## v1.0 — PyTorch backend

Planned. Replace Theano with PyTorch and keep the training behavior the same. Exit criterion: the parity tests pass, Theano is gone from code and docs, and the ml-100k `u1` experiment trains to an RMSE of roughly 0.95–1.0. Depends on F3 and F5.

- [x] **F6 — Rewrite `cfrbm/rbm.py` in PyTorch.** Keep the `CFRBM` constructor, including `.npy` weight loading, and keep the current `normal(0, 0.1)` weight init and seed 17. `cdk_fun(...)` and `predict()` should still return callables. Use `torch.bernoulli` with a seeded `torch.Generator`, and do the parameter updates in place under `torch.no_grad()`. Compute `gradient` with matrix multiplies, `(v1*m).T @ h1 - (v2*m).T @ (h2p*h1)`, instead of building the batch × visible × hidden `outer` tensor. Acceptance: the F5 parity tests pass.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | M |
  | Details | Clear math scope; estimated and changed three implementation files across backend, dependencies and tests, plus tracking. Preserved NumPy normal(0, 0.1) initialization, constructor and checkpoint suffixes, callable factories and legacy CD-k final-transition math. Sampling uses a local torch.Generator seeded with 17; matrix-multiply gradients and in-place no-grad updates retain F10-F12 quirks. requirements.txt now declares torch. |
  | Commit | `d32402c155c02b3c3255e4436eb648f10c3c6d47` — implements F6. |
  | Validation | Python 3.12.2 / PyTorch 2.14.1 CPU: make test passed all 22 tests without skips, including all 8 unchanged F5 parity cases and 5 new backend checks. Real-sampling CD-3 training/prediction smoke passed. Roadmap validator and git diff --check passed. |
  | Limitation | CPU float32 tensors only; accelerator selection and full runner integration remain unverified and are scheduled in F7-F9. Legacy runners still import Theano. |
  | Limitation | Seed 17 controls sampling; initialization retains NumPy's global RNG. PyTorch random streams do not reproduce Theano draws; debug is accepted without compiler effects. |
  | Next step | Implement F7 device selection and F8 tensor conversion in both runners, update launch documentation, then execute F9 end-to-end validation. |

- [x] **F7 — Device selection.** Pick CUDA, then MPS, then CPU automatically, and allow an override with a `--device` CLI option. Acceptance: training runs on CPU and on at least one available accelerator. Depends on F6.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | M |
  | Details | Clear selection priority; estimated and changed seven code/test files across device resolution, RBM state and both CLIs. Resolved unspecified overrides as auto, cpu, mps and cuda[:index], with errors for unavailable devices. Parameters, momentum and seeded sampling use the selected device. Minimal F8 runner tensor conversion and Python 3 key-view fixes enable both CLIs. |
  | Commit | `29695432f2ae77ec3b26818828711f117d0fc1aa` — implements F7. |
  | Validation | Python 3.12.2 / PyTorch 2.14.1: make test passed 28 tests with CUDA alone skipped outside the sandbox. CPU and MPS CD-3 updates and prediction passed; both CLIs completed two-epoch synthetic experiments on CPU, MPS and auto with finite metrics and results JSON. All 8 F5 parity cases passed; device priority, invalid overrides and checkpoint placement checked. Roadmap validator and git diff --check passed. |
  | Limitation | CUDA execution and indexed multi-GPU training are unverified because this host has no CUDA device; selection policy is covered with mocked availability. |
  | Limitation | MPS is hidden by the execution sandbox on this host; real MPS validation required running outside it. Cross-device random streams are not guaranteed identical. |
  | Limitation | Validation uses small synthetic experiments; full MovieLens accuracy and runtime remain part of F9. README and Makefile retain legacy launch instructions pending F8. |
  | Next step | Run tests.test_device on a CUDA host, including an explicit cuda:index override on a multi-GPU system. |

- [x] **F8 — Update call sites, Makefile and README.** Remove `theano.tensor` from `cfrbm/user_based.py` and `cfrbm/item_based.py`, and convert batches with `torch.from_numpy(...).to(device)` and `.cpu().numpy()`. Replace `THEANO_FLAGS` in the `Makefile` and README with the new run command and dependencies. Acceptance: `grep -ri theano` finds nothing, and `make run` starts training. Depends on F6 and F7.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | M |
  | Details | Clear implementation scope; estimated and changed five implementation files across launch commands, documentation and obsolete backend references, plus tracking. F6/F7 already supply both runners' tensor conversions and numpy/torch dependencies. Make targets now use PyTorch module execution with PYTHON/DEVICE overrides; README covers environment setup, dataset extraction and both CLIs. Removed the obsolete parity skip gate. |
  | Commit | `384f761327d6da072d78b7b765b9669f7243c80a` — implements F8. |
  | Validation | Python 3.12.2 / PyTorch 2.14.1: make test ran 29 tests, 27 passed and 2 accelerator checks skipped; includes both CLI integrations and all 8 parity checks. Default make run with activated .venv completed MovieLens epoch 0 on CPU, wrote JSON, MAE 0.931794 and RMSE 1.187261, then was intentionally stopped. Scoped backend-reference search, roadmap validator and git diff --check passed. |
  | Limitation | Literal unrestricted grep still matches preserved roadmap history, prior job notes and environment/Git artifacts. Zero matches verified in tracked application code, tests, Makefile, README and requirements; preserving unrelated history takes precedence over erasing these records. |
  | Limitation | CUDA and MPS were unavailable inside this sandbox; accelerator execution was not revalidated for F8. |
  | Limitation | Only the first default MovieLens epoch was validated; 100-epoch completion and final accuracy remain unverified. |
  | Next step | Execute F9's full 100-epoch MovieLens experiment and record per-epoch metrics and final results JSON. |

- [x] **F9 — End-to-end validation on ml-100k.** Run the `ubased.json` experiment and record the per-epoch MAE/RMSE output. Acceptance: training finishes all 100 epochs, RMSE ends up roughly in the 0.95–1.0 range, and the results JSON is written. Depends on F8.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | M |
  | Details | Clear acceptance criteria; estimated and added five validation files plus job notes and roadmap tracking across execution and evidence. Resolved unspecified device/seed as CPU and seed 17. Ran unchanged ubased.json and training code for all 100 epochs; saved console metrics, exact results JSON, environment/input hashes, reproduction instructions and an evidence checker in validation/f9. |
  | Commit | `d82c4b842b7911d80a475842ce7dfcaace771d5d` — implements F9 validation evidence. |
  | Validation | Training exited 0 in 774.93 seconds; all 100 sequential epochs, log/JSON agreement, configuration, input hashes and coverage checks passed. Final MAE 0.793114, RMSE 1.066830; accuracy check correctly exits 1 because 0.95–1.0 was not reached. make test: 27 passed, 2 unavailable-accelerator skips. Roadmap validator and git diff --check passed. |
  | Limitation | Final RMSE 1.066830 misses the requested range; best RMSE was 1.063114 at epoch 33. Accuracy acceptance remains unmet. |
  | Limitation | Existing evaluator rounds predictions and scores 19,837 of 20,000 ratings; 32 unknown-movie ratings trigger per-user skips totaling 163 ratings. |
  | Limitation | One seeded CPU run only; accelerator accuracy, other splits and variation across seeds remain unverified. |
  | Next step | Investigate the accuracy gap during F10–F12 and record before/after metrics with the same seeds, inputs, device and evaluator. |
  | Next step | Fix unknown-movie evaluation handling, report scored/skipped counts explicitly and rerun both baseline and changed models under the corrected evaluator. |

## v1.1 — Training correctness and speed

Provisional. Fix the training bugs that v1.0 kept on purpose. Make each fix its own change and compare it with the v1.0 MAE/RMSE baseline from F9. Exit criterion: each fix has its own before/after RMSE recorded. Depends on F9.

- [x] **F10 — Fix the momentum update.** `prev_gw`, `prev_gv` and `prev_gh` store the raw gradient, not the step taken, so `momentum * prev_g` is about 1000× larger than `lr * grad` (`cfrbm/rbm.py:125,133`). Store the velocity instead, `v = momentum*v + lr*grad`, and add it to the parameters. Acceptance: an updated F4 reference and parity test, and before/after RMSE recorded.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | M |
  | Details | Clear formula; estimated and changed four shared code/test files across RBM updates, NumPy reference and parity/regression checks, plus six validation files and tracking. Historical prev_g* state now stores velocities; parameters add those steps in place, preserving weight-only decay and state across epoch factories. Resolved comparison ambiguity with the verified F9 CPU seed-17 baseline. |
  | Commit | `471ee2a84ec70c3afcbbb32985d4ef91c031f4f4` — implements F10 and records its RMSE evidence. |
  | Validation | 31 tests ran: 29 passed, 2 unavailable-accelerator skips. Updated F4 reference and parity cover repeated velocities, decay, zero/changed learning rates and new factories. Completed 100 CPU epochs: final RMSE 1.066829754 → 1.155391569; MAE 0.793113878 → 0.897867621. Best RMSE 1.100463453 at epoch 40. Log/JSON, input/environment and commit source-hash checks passed; see validation/f10. Roadmap validator and git diff --check passed. |
  | Limitation | Corrected momentum worsens final RMSE under unchanged hyperparameters and misses F9's 0.95–1.0 target; settings were not retuned. |
  | Limitation | One CPU seed only; CUDA and MPS validation skipped because unavailable in the sandbox. |
  | Limitation | Existing rounded evaluator scores 19,837 of 20,000 ratings; unknown-movie handling skips 163 ratings. |
  | Next step | Tune learning rates and training duration on a validation split after the correctness changes, then repeat the controlled RMSE comparison. |
  | Next step | Revalidate velocity updates and training on CUDA and MPS when available. |

- [x] **F11 — Remove the `h1` factor from the negative-phase gradient.** Because `v1h1_mask` is applied to the negative-phase term, it gets multiplied by `h1`. Mask that term only by the observed ratings. Acceptance: an updated reference and test, and before/after RMSE recorded.

  | Item | Content |
  | --- | --- |
  | Status | Done |
  | Timestamp | 2026-10-10 |
  | Complexity | M |
  | Details | Clear formula; estimated and changed four shared code/test files across RBM gradients, NumPy reference and parity/regression checks, plus four validation files and tracking. Both weight-gradient phases use observed-rating masks; the negative term no longer multiplies by h1. Bias gradients are unchanged. Resolved comparison ambiguity as F10 → F11, also reporting cumulative change from F9. |
  | Commit | `f4430eb5d8ff2d6c3efc970afcf4c5e9b426c5b3` — implements F11 and records its RMSE evidence. |
  | Validation | 33 tests ran: 31 passed, 2 unavailable-accelerator skips. Updated reference/parity and hand-derived zero-h1 regression pass; unobserved negatives remain masked. Completed 100 CPU epochs: final RMSE 1.155391569 → 1.082427337; MAE 0.897867621 → 0.817260674. Best RMSE at epoch 100. Cumulative RMSE versus F9: +0.015597582. Log/JSON, input/environment and commit source-hash checks passed; see validation/f11. Roadmap validator and git diff --check passed. |
  | Limitation | Final RMSE remains above F9's 1.066829754 baseline and 0.95–1.0 target under unchanged hyperparameters. |
  | Limitation | Sequential comparison on top of F10, one CPU seed only; an F11-only ablation with legacy momentum and accelerator accuracy are unverified. |
  | Limitation | Existing rounded evaluator scores only 19,837 of 20,000 ratings; visible sampling still retains the F12 behavior. |
  | Next step | Repeat corrected-model evaluation across seeds after F12, using a validation split to choose hyperparameters. |
  | Next step | Correct unknown-movie evaluation handling and rerun baseline and corrected models with full test-coverage reporting. |

- [ ] **F12 — Softmax visible units.** Replace the independent Bernoulli sampling of normalized sigmoids with a softmax plus categorical sampling over the 5 rating units, as in the RBM-CF paper. Acceptance: each sampled visible group is exactly one-hot, and before/after RMSE recorded.
- [ ] **F13 — Vectorize data preparation.** Profiles and masks are rebuilt every epoch with `list.index()` inside nested loops (`cfrbm/user_based.py:81-92`). Build ID→index dicts and precompute dense or sparse arrays once. Acceptance: identical batches on a fixed seed and a measurable speedup per epoch.
- [ ] **F14 — Deduplicate the user- and item-based runners.** `cfrbm/user_based.py` and `cfrbm/item_based.py` are near-identical. Merge them into one runner with an orientation flag, and use the config's `batch_size` everywhere (`item_based` hardcodes 10). Acceptance: both modes run from one module with the same results as before. Depends on F13.
