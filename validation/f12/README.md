# F12 — Softmax visible units

Each consecutive group of five visible logits now uses softmax, followed by
one categorical draw using `torch.multinomial` and the model's local generator.
The resulting visible sample is exactly one-hot in every group, including
groups subsequently excluded by gradient masks. Hidden units remain Bernoulli.
This implements the visible conditional distribution in equation 1 of
[Salakhutdinov, Mnih and Hinton (2007)](https://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf).

`prop_down(h, k=5)` now returns grouped softmax probabilities directly. The
existing `sample_visible(h, k=5)` interface returns samples and probabilities
with the same shape, float32 dtype and device as before. Invalid group widths
are rejected before reshaping, so groups cannot span different users.
`predict()` remains stochastic through its hidden sample and returns softmax
probabilities. Weight loading, gradients, momentum and CD-k behavior are
unchanged. Random streams differ from the old independent Bernoulli draws.

## Outcome — 2026-10-10

All 100 epochs completed successfully in 778.37 seconds. Final RMSE changed
from **1.082427337** (F11) to **1.088789011** (F12), a regression of
**0.006361675** under unchanged hyperparameters. Final MAE changed from
**0.817260674** to **0.819982860**. Best F12 RMSE was **1.085798560** at epoch 99.

The cumulative F9-to-F12 RMSE change is **+0.021959257** (F9: 1.066829754).
The corrected sampler satisfies the one-hot requirement but does not improve
accuracy in this run or reach F9's 0.95–1.0 target. The evidence verifier
confirmed all 100 sequential epochs, finite metrics, log/JSON agreement,
matching inputs, seeds, environment and unchanged evaluation code. Captured
application source hashes also match the implementation.

## Experiment protocol

Compare F12 against F11 to measure this change on top of F10/F11, and against
F9 for the cumulative difference. The F11 evidence was verified against
implementation commit `f4430eb5d8ff2d6c3efc970afcf4c5e9b426c5b3` before changing
the model; the working application sources matched that commit exactly.

All runs use unchanged `ubased.json`, 100 epochs, MovieLens u1, CPU,
NumPy/sampling/hash seed 17, one CPU thread, identical initial weights and the
existing evaluator. The source snapshot isolates the training run from edits
and preserves the existing root results file.

```sh
.venv/bin/python validation/run_training.py f12
.venv/bin/python validation/verify_training.py f12
```

Use a disposable reproduction checkout with dependencies and dataset present;
relocate `validation/f12/run.json` first because capture refuses to overwrite
existing evidence. Supply the F12 implementation commit from ROADMAP.md as the
verifier's second argument to compare all captured application source hashes
to Git. The evidence format is described in [the F10 record](../f10/README.md).

## Validation and limits

The full suite ran 36 tests: 34 passed and two accelerator tests were skipped
in the sandbox. Outside the sandbox, the seven device tests passed on CPU
and MPS (six passes, only CUDA skipped), including categorical one-hot samples,
CD-3 training and both CLI orientations.

The NumPy reference now uses max-shifted exponentials and the parity harness
injects categorical indices as well as hidden Bernoulli samples. Regression
checks cover hand-derived probabilities, large positive/negative logits,
invariance to group offsets, invalid group widths, 40,000 one-hot rating
groups, empirical categorical frequencies, local RNG isolation/replay,
prediction, masked gradients and CD-2 updates. Device tests check one-hot
samples and CD-3 training on every available backend; both CLI orientations
also complete small end-to-end runs.

The RMSE comparison covers one CPU seed and the sequential F11-to-F12 change.
It uses rounded predictions and scores 19,837 of 20,000 test ratings; existing
unknown-movie handling skips 163 ratings. Other seeds, full-coverage accuracy
and accelerator MovieLens accuracy are unverified. Tune hyperparameters on a
validation split and rerun baseline and corrected models after fixing test
coverage. This feature changes the visible conditional distribution; it does
not implement the paper's full prediction or missing-rating algorithm.
