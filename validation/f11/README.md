# F11 — Observed-only negative-phase mask

The weight gradient is now
`((v1 * masks).T @ h1 - (v2 * masks).T @ h2p) / batch_size`.
A zero positive hidden sample no longer suppresses negative-phase hidden
probabilities. Both visible phases still use the observed-rating mask;
visible- and hidden-bias gradient formulas are unchanged.

## Outcome — 2026-10-10

All 100 epochs completed successfully in 1,550.93 seconds. Relative to F10,
final RMSE improved from **1.155391569** to **1.082427337** (-0.072964232),
and final MAE improved from **0.897867621** to **0.817260674**. Best F11 RMSE
was **1.082427337** at epoch 100. The evidence verifier passes all checks.

The cumulative F9-to-F11 RMSE change is **+0.015597582** (F9: 1.066829754).
Thus the corrected model still misses F9's 0.95–1.0 target under unchanged
hyperparameters. Complete F12 and evaluate learning-rate/training-schedule
choices on a validation split; keep full test-coverage correction separate
and rerun comparisons when that evaluator changes.

## Experiment protocol

F11 is applied on top of F10. Compare the final score with F10 to isolate this
change, and with F9 to report the cumulative difference. All runs use unchanged
`ubased.json`, 100 epochs, MovieLens u1, CPU, NumPy/sampling/hash seed 17, one CPU
thread, identical initial weights and the existing evaluator.

```sh
.venv/bin/python validation/run_training.py f11
.venv/bin/python validation/verify_training.py f11
```

Use a disposable reproduction checkout with dependencies and dataset present;
relocate `validation/f11/run.json` first because capture refuses to overwrite
existing evidence. The capture/verification format is described in
[the F10 record](../f10/README.md). The source hashes identify the exact code
used by the isolated run even though it started before the implementation
commits were created. Supply the F11 implementation commit from ROADMAP.md as
the verifier's second argument to compare every application source hash to Git.

## Validation and limits

The combined suite runs 33 tests: 31 pass, with CUDA and MPS skipped because
they are unavailable in the sandbox. The independent outer-product NumPy
reference agrees with matrix-multiply backend gradients and CD updates.
A hand-derived zero-h1 case rejects the original extra hidden factor; changing
unobserved negative values while holding h2p fixed leaves gradients unchanged.
Existing cases cover mixed masks, a fully masked user, hidden biases and CD-2.

This is a sequential F10-to-F11 comparison, not an F11-only ablation on legacy
momentum. Results cover one CPU seed and 19,837 of 20,000 ratings with rounded
predictions; the existing unknown-movie handling skips 163 ratings. Other
seeds, accelerators and full-coverage accuracy remain unverified. Visible
sampling is still independent Bernoulli over normalized sigmoids (F12).
