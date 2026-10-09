# F10 — Velocity momentum

`prev_gw`, `prev_gv` and `prev_gh` retain the parameter step:
`velocity = momentum * velocity + learning_rate * gradient`.
Weight decay remains inside the weight gradient before learning-rate scaling;
biases are not decayed. Existing state names and tensor identities are retained,
and velocities survive creation of a new training callable each epoch.

## Outcome — 2026-10-10

All 100 epochs completed successfully in 1,547.62 seconds. Final RMSE changed
from F9's **1.066829754** to **1.155391569** (+0.088561814); final MAE changed
from **0.793113878** to **0.897867621**. Best F10 RMSE was **1.100463453** at
epoch 40. The evidence verifier passes all integrity checks.

Correcting momentum worsens accuracy under the unchanged hyperparameters;
this does not meet F9's 0.95–1.0 target. Retune learning rates and the training
schedule after the correctness changes, measuring on a validation split.
The independent Bernoulli visible sampler and negative-phase hidden factor
remain in this F10-only run; F11 is evaluated separately on top of F10.

## Experiment protocol

Compare with the committed F9 baseline: the unchanged `ubased.json`, 100 epochs,
MovieLens u1, CPU, NumPy and sampling seed 17, `PYTHONHASHSEED=17`, and one CPU
thread. The initial weights, data ordering, hyperparameters and evaluator are
unchanged. F10 retains the negative-phase hidden factor until F11.

From a clean checkout of this implementation with the dataset and dependencies
installed, capture a new run using:

```sh
.venv/bin/python validation/run_training.py f10
.venv/bin/python validation/verify_training.py f10
```

The capture script refuses to overwrite existing evidence; remove or relocate
only `validation/f10/run.json` in a disposable reproduction checkout first.
It copies application sources and config into a temporary directory, links the
dataset, and runs the real CLI. It never overwrites root experiment results.
`run.json` records input/source hashes, versions, seeds, timings and exit code;
`training.log` and the results JSON retain all per-epoch metrics. The parent
commit identifies the starting tree; source SHA-256 hashes identify the exact
uncommitted implementation evaluated. Pass the implementation commit from
ROADMAP.md as the verifier's second argument to check those source hashes.

## Validation and limits

The F10 test suite runs 31 tests: 29 pass, CUDA and MPS checks skip because
neither device is available in the sandbox. Hand-calculated reference and
backend parity tests cover scaled velocity memory, repeated updates, zero
learning rates, changed learning rates/momentum, decay and new epoch factories.
The existing CD-2 check verifies in-place state and no autograd tracking.

This is one CPU seed with the existing rounded evaluator: 19,837 of 20,000
ratings scored. Unknown movies trigger per-user skips totaling 163 ratings.
It does not establish results on all test ratings, other seeds or accelerators.
