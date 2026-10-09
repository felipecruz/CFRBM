# Auto Pilot

Auto Pilot execution summaries appear here.


## 2026-10-09T22:37:16Z — F1
Job `DD279C68-8DF4-42EC-B8D7-643AC1DF2448` · Done
- F1 — Port syntax to Python 3.: Done. Validation: Python 3.12.2: make test passed all 4 existing tests. Isolated AST checks of both get_index functions passed integer-type, schedule-boundary, and fallback cases. Roadmap validator and git diff --check passed..
  Summary: Clear specification; estimated three Python files across utilities and two runners, plus roadmap tracking. Replaced xrange and used integer division in the reshape and both levels of schedule indexing, preserving Python 2 behavior.
  ⚠ Limitation: Full training remains unverified; legacy Theano dependencies and runner imports are unchanged. Runner dictionary-key views remain incompatible with chunker's sequence slicing on Python 3.
  ⚠ Next step: Convert runner key views to sliceable sequences or adapt chunker before validating full Python 3 training.
Notes: [.aiflow/notes/NOTES_DD279C68-8DF4-42EC-B8D7-643AC1DF2448.MD](.aiflow/notes/NOTES_DD279C68-8DF4-42EC-B8D7-643AC1DF2448.MD)


## 2026-10-09T22:38:32Z — F2
Job `61A49F78-C352-4017-A10A-2383A5219C58` · Done
- F2 — Fix item-based experiment loading.: Done. Validation: Python 3.12.2: isolated entry-point checks reproduced the original TypeError, then passed with the real ubased.json reader, multiple experiments/configurations, and an empty list; dataset loading and training were mocked. make test passed all 4 tests. Roadmap validator and git diff --check passed..
  Summary: Clear specification with no ambiguities; estimated one Python file in the item-based runner, plus roadmap tracking. The runner now loops over experiments and their configurations like user_based, retaining item-based dataset orientation.
  ⚠ Limitation: Full CLI training remains unverified because of existing legacy Theano/import dependencies and Python 3 dictionary-key slicing issues.
  ⚠ Next step: After the existing dependency/import and key-view compatibility work, validate item-based training end to end.
Notes: [.aiflow/notes/NOTES_61A49F78-C352-4017-A10A-2383A5219C58.MD](.aiflow/notes/NOTES_61A49F78-C352-4017-A10A-2383A5219C58.MD)


## 2026-10-09T22:40:20Z — F3
Job `2573189E-7E0F-4DF0-9A93-CBFD52D6AC9C` · Done
- F3 — Make the package importable and declare dependencies.: Done. Validation: Python 3.12.2: make test passed all 4 tests. With only Theano stubbed, all six cfrbm modules imported and both module entry points parsed ubased.json and reached dataset loading. make -n run, roadmap validator and git diff --check passed..
  Summary: Clear import scope; estimated five implementation files across two runners, dependencies and launch documentation, plus tracking. Both runners use package-relative imports; README and Makefile use module execution. requirements.txt declares numpy and the current Theano backend, interpreting torch from v1.0 as a future dependency.
  ⚠ Limitation: The real user_based CLI still fails during Theano import because collections.MutableMapping is unavailable on Python 3.12; the full import acceptance check is not met in this environment. Training remains unverified; the configured ml-100k/u1.base file is absent.
  ⚠ Next step: During F6/F8, replace Theano with torch in requirements.txt and rerun unmocked package imports and both CLI entry points on Python 3.12.
Notes: [.aiflow/notes/NOTES_2573189E-7E0F-4DF0-9A93-CBFD52D6AC9C.MD](.aiflow/notes/NOTES_2573189E-7E0F-4DF0-9A93-CBFD52D6AC9C.MD)


## 2026-10-09T22:42:47Z — F4
Job `F8B3BF21-65F3-4D49-98A8-6E901841F03F` · Done
- F4 — NumPy reference implementation of the CD math.: Done. Validation: Python 3.12.2: make test passed all 9 tests, including 5 new hand-calculated regression tests covering propagation, two rating groups per user, masks, momentum, decay, defaults, deterministic updates and unchanged inputs. Roadmap validator and git diff --check passed..
  Summary: Clear math scope; estimated and changed two test files across a NumPy reference and deterministic regression tests, plus tracking. Resolved sampling ambiguity with supplied binary visible samples. Added propagation, grouped sigmoid normalization, masked gradients and simultaneous float32 updates; preserved raw-gradient momentum, the negative-phase 
  ⚠ Limitation: Reference accepts fixed samples and does not reproduce random-number streams; live Theano numerical parity remains unverified on Python 3.12.
  ⚠ Next step: In F5, compare the backend against these reference functions using fixed inputs and samples.
Notes: [.aiflow/notes/NOTES_F8B3BF21-65F3-4D49-98A8-6E901841F03F.MD](.aiflow/notes/NOTES_F8B3BF21-65F3-4D49-98A8-6E901841F03F.MD)


## 2026-10-09T22:46:19Z — F5
Job `361B69F1-5276-449F-8025-F2EF07C60E0D` · Done
- F5 — Parity test harness.: Done. Validation: Python 3.12.2: make test passed 9 tests with the PyTorch parity class explicitly skipped. A temporary NumPy adapter executing legacy math passed all 8 harness checks and rejected 5 injected faults. Confirmed missing torch fails all 8 cases after removing the legacy import gate. Roadmap validator and git diff --check passed..
  Summary: Clear math scope; estimated and changed one test file across fixtures, backend adaptation and parity assertions, plus tracking and handoff notes. Eight reusable checks compare propagation, fixed samples, five-way normalization, masked gradients and repeated CD-1 updates against F4 with shape checks and allclose. Resolved future API ambiguity in an 
  ⚠ Limitation: Actual PyTorch adapter execution and backend parity remain unverified until F6; the current Theano implementation is explicitly skipped and torch is not installed. The harness covers CD-1 updates, not CD-k chains or device-to-device parity.
  ⚠ Next step: In F6, run all 8 parity cases without skips; if the callable factory signature changes, update TorchAdapter.train while retaining the reference checks.
Notes: [.aiflow/notes/NOTES_361B69F1-5276-449F-8025-F2EF07C60E0D.MD](.aiflow/notes/NOTES_361B69F1-5276-449F-8025-F2EF07C60E0D.MD)


## 2026-10-09T22:52:02Z — F6
Job `C056C23A-026B-4849-9763-8D49670CF436` · Done
- F6 — Rewrite cfrbm/rbm.py in PyTorch.: Done. Validation: Python 3.12.2 / PyTorch 2.14.1 CPU: make test passed all 22 tests without skips, including all 8 unchanged F5 parity cases and 5 new backend checks. Real-sampling CD-3 training/prediction smoke passed. Roadmap validator and git diff --check passed..
  Summary: Clear math scope; estimated and changed three implementation files across backend, dependencies and tests, plus tracking. Preserved NumPy normal(0, 0.1) initialization, constructor and checkpoint suffixes, callable factories and legacy CD-k final-transition math. Sampling uses a local torch.Generator seeded with 17; matrix-multiply gradients and in
  ⚠ Limitation: CPU float32 tensors only; accelerator selection and full runner integration remain unverified and are scheduled in F7-F9. Legacy runners still import Theano. Seed 17 controls sampling; initialization retains NumPy's global RNG. PyTorch random streams do not reproduce Theano draws; debug is accepted without compiler effects.
  ⚠ Next step: Implement F7 device selection and F8 tensor conversion in both runners, update launch documentation, then execute F9 end-to-end validation.
Notes: [.aiflow/notes/NOTES_C056C23A-026B-4849-9763-8D49670CF436.MD](.aiflow/notes/NOTES_C056C23A-026B-4849-9763-8D49670CF436.MD)


## 2026-10-09T22:55:14Z — F7
Job `780CA687-6796-4357-9E6E-E6A3812312A9` · Done
- F7 — Device selection.: Done. Validation: Python 3.12.2 / PyTorch 2.14.1: make test passed 28 tests with CUDA alone skipped outside the sandbox. CPU and MPS CD-3 updates and prediction passed; both CLIs completed two-epoch synthetic experiments on CPU, MPS and auto with finite metrics and results JSON. All 8 F5 parity cases passed; device priority, invalid overrides and checkpoint placemen.
  Summary: Clear selection priority; estimated and changed seven code/test files across device resolution, RBM state and both CLIs. Resolved unspecified overrides as auto, cpu, mps and cuda[:index], with errors for unavailable devices. Parameters, momentum and seeded sampling use the selected device. Minimal F8 runner tensor conversion and Python 3 key-view f
  ⚠ Limitation: CUDA execution and indexed multi-GPU training are unverified because this host has no CUDA device; selection policy is covered with mocked availability. MPS is hidden by the execution sandbox on this host; real MPS validation required running outside it. Cross-device random streams are not guaranteed identical. Validation uses small synthetic exper
  ⚠ Next step: Run tests.test_device on a CUDA host, including an explicit cuda:index override on a multi-GPU system.
Notes: [.aiflow/notes/NOTES_780CA687-6796-4357-9E6E-E6A3812312A9.MD](.aiflow/notes/NOTES_780CA687-6796-4357-9E6E-E6A3812312A9.MD)


## 2026-10-09T22:57:39Z — F8
Job `BE6944FB-6BCF-4772-9F89-8F3E987FA320` · Done
- F8 — Update call sites, Makefile and README.: Done. Validation: Python 3.12.2 / PyTorch 2.14.1: make test ran 29 tests, 27 passed and 2 accelerator checks skipped; includes both CLI integrations and all 8 parity checks. Default make run with activated .venv completed MovieLens epoch 0 on CPU, wrote JSON, MAE 0.931794 and RMSE 1.187261, then was intentionally stopped. Scoped backend-reference search, roadmap val.
  Summary: Clear implementation scope; estimated and changed five implementation files across launch commands, documentation and obsolete backend references, plus tracking. F6/F7 already supply both runners' tensor conversions and numpy/torch dependencies. Make targets now use PyTorch module execution with PYTHON/DEVICE overrides; README covers environment se
  ⚠ Limitation: Literal unrestricted grep still matches preserved roadmap history, prior job notes and environment/Git artifacts. Zero matches verified in tracked application code, tests, Makefile, README and requirements; preserving unrelated history takes precedence over erasing these records. CUDA and MPS were unavailable inside this sandbox; accelerator execut
  ⚠ Next step: Execute F9's full 100-epoch MovieLens experiment and record per-epoch metrics and final results JSON.
Notes: [.aiflow/notes/NOTES_BE6944FB-6BCF-4772-9F89-8F3E987FA320.MD](.aiflow/notes/NOTES_BE6944FB-6BCF-4772-9F89-8F3E987FA320.MD)


## 2026-10-09T23:12:39Z — F9
Job `B65AE4FC-5A3F-406B-A9CA-E7A295892595` · Done
- F9 — End-to-end validation on ml-100k.: Done. Validation: Training exited 0 in 774.93 seconds; all 100 sequential epochs, log/JSON agreement, configuration, input hashes and coverage checks passed. Final MAE 0.793114, RMSE 1.066830; accuracy check correctly exits 1 because 0.95–1.0 was not reached. make test: 27 passed, 2 unavailable-accelerator skips. Roadmap validator and git diff --check passed..
  Summary: Clear acceptance criteria; estimated and added five validation files plus job notes and roadmap tracking across execution and evidence. Resolved unspecified device/seed as CPU and seed 17. Ran unchanged ubased.json and training code for all 100 epochs; saved console metrics, exact results JSON, environment/input hashes, reproduction instructions an
  ⚠ Limitation: Final RMSE 1.066830 misses the requested range; best RMSE was 1.063114 at epoch 33. Accuracy acceptance remains unmet. Existing evaluator rounds predictions and scores 19,837 of 20,000 ratings; 32 unknown-movie ratings trigger per-user skips totaling 163 ratings. One seeded CPU run only; accelerator accuracy, other splits and variation across seeds
  ⚠ Next step: Investigate the accuracy gap during F10–F12 and record before/after metrics with the same seeds, inputs, device and evaluator. Fix unknown-movie evaluation handling, report scored/skipped counts explicitly and rerun both baseline and changed models under the corrected evaluator.
Notes: [.aiflow/notes/NOTES_B65AE4FC-5A3F-406B-A9CA-E7A295892595.MD](.aiflow/notes/NOTES_B65AE4FC-5A3F-406B-A9CA-E7A295892595.MD)
