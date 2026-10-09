# F9 — MovieLens 100k baseline

This directory records a full run of the unchanged `ubased.json` experiment
through `cfrbm.user_based` on CPU. The training implementation and configuration
are unchanged from the F8 baseline identified by `source_commit` in `run.json`.

## Outcome — 2026-10-10

Training exited successfully after all 100 epochs in 774.93 seconds (12m 55s).
The results JSON was written and all 100 console records match it exactly.
Final MAE is **0.793113878** and final RMSE is
**1.066829754**. Best RMSE was **1.063113927** at
iteration 32 (epoch 33). **The requested final RMSE range of 0.95–1.0
was not met.** F9 records a completed validation run, not a successful accuracy
acceptance result.

`verify.py` passes the artifact-integrity and coverage checks, then deliberately
exits with status 1 for the unmet RMSE criterion. The existing test suite ran
29 tests: 27 passed and 2 accelerator tests were skipped because CUDA and MPS
were unavailable in the sandbox. The roadmap validator passed.

Next: investigate the accuracy gap through the planned F10–F12 correctness
changes, recording before/after scores under the same evaluation conditions.
Separately, fix the evaluator's handling of unknown movies and report coverage
explicitly before claiming a score on the entire test split.

## Evidence

- `training.log`: complete stdout/stderr, including MAE/RMSE for iterations 0–99
  (100 epochs).
- `initial_experiment_100k-u1xu1.json`: an exact copy of the results JSON written
  by the runner in the repository root.
- `run.json`: command, environment, source commit, input SHA-256 hashes,
  dependency versions, device, seeds, timestamps, exit code and elapsed time.
- `verify.py`: checks successful exit, input hashes, unchanged configuration,
  all 100 sequential epochs, finite metrics, the momentum schedule, exact
  console/JSON agreement, and final RMSE within 0.95–1.0.

## Reproduce

Install the dependencies and extract `ml-100k` as described in the root README.
Run from the repository root using the Python/NumPy/PyTorch versions recorded
in `run.json`. The wrapper seeds NumPy before executing the real module entry
point; `PYTHONHASHSEED` fixes the movie-column order produced from a set.
The RBM's existing sampling generator uses seed 17. One CPU thread avoids
threading overhead for small batches.

```sh
PYTHONHASHSEED=17 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u -c 'import runpy, sys, numpy as np; np.random.seed(17); sys.argv = ["cfrbm.user_based", "ubased.json", "--device", "cpu"]; runpy.run_module("cfrbm.user_based", run_name="__main__")'
```

This writes `initial_experiment_100k-u1xu1.json` in the repository root,
overwriting any prior result there. The committed copy in this directory
remains the original F9 baseline. Exact numeric reproduction across library
versions, Python versions or devices is not guaranteed.

Check the captured baseline, with the original dataset present:

```sh
.venv/bin/python validation/f9/verify.py
make test PYTHON=.venv/bin/python
```

## Evaluation limits

The training split contains 80,000 ratings, 943 users and 1,650 movies. The
current evaluator scores 19,837 of the 20,000 test ratings (99.185%). There are
32 ratings for movies absent from training; the runner's broad exception
handler also skips the remaining ratings for a user after their first unknown
movie, bringing the total skipped to 163. These counts follow the test-file
order and the existing evaluation loop. Predictions are rounded to integer
ratings before computing MAE/RMSE.

This is a single seeded CPU baseline of the existing implementation. It does
not establish accuracy on all 20,000 ratings, other splits, multiple seeds or
accelerators. The legacy update and sampling behavior targeted by F10–F12 is
preserved. Compare those changes against this baseline with the same seeds,
input hashes, device and evaluation behavior. If evaluation coverage is fixed,
rerun the original and changed models under the same corrected evaluator.
