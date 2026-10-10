# F13 — Cached data preparation

Both runners now build ID-to-column and profile-to-row dictionaries and dense
float32 profiles/masks once per `run`. Training uses contiguous views of the
cache; evaluation gathers rows in test order. NumPy constructs the five-way
one-hot array in one operation. Rating ingestion remains a single linear pass
so repeated ratings retain last-write-wins behavior.

Profile order follows the training file; visible order follows the caller's
ID list. The cache does not consume random numbers. Test-only profiles have
zero rows and join training after the first evaluation, matching the previous
`defaultdict` side effect. Unknown evaluation IDs still stop scoring the rest
of that profile. These behaviors are preserved for batch and RNG parity.

## Reproduce

From the repository root, with `.venv` dependencies and MovieLens u1 installed:

```sh
make test PYTHON=.venv/bin/python
PYTHONHASHSEED=17 PYTHONPATH=. .venv/bin/python validation/benchmark_preparation.py > /tmp/f13-benchmark.json
```

The benchmark loads the original runners from baseline commit
`34a0ce7b2419fca20fef771afe05232506603b3f`. It checks every training and test
batch over two epochs against an independent copy of the old preparation,
then alternates three measurements of each preparation path after warmup.
Preparation timings include training profiles/masks and test profiles/masks
(the old path needlessly built evaluation masks); they exclude model execution,
file parsing and device transfer. Cache construction is timed separately.

The script also runs both original and changed runners for three MovieLens
epochs on CPU with one Torch thread and NumPy/hash/sampling seed 17. It uses
`ubased.json` with only the epoch count and temporary result destination changed.
It checks exact metric equality, combined training/mask/prediction-input hashes,
all six final model-state hashes and final RNG-state hashes. Training timings
include setup, training, evaluation, JSON output and identical batch hashing;
they are single short runs rather than a statistically controlled throughput
study. All output files are temporary except the requested benchmark JSON.

## Results — 2026-10-10

All batch comparisons and all three-epoch runner comparisons passed exactly
for both orientations. Full test suite: 39 tests, 37 passed and two unavailable
accelerator skips. The three new regression tests cover ordering, repeated IDs
and ratings, rounding, zero masks, empty inputs, incomplete batches, test-only
profiles, cache reuse, RNG isolation and seeded CD-1/CD-2/CD-3 runner replay.

| Measurement | User-based | Item-based |
| --- | --- | --- |
| Median legacy preparation per epoch | 6.747701 s | 5.410540 s |
| Median cached preparation per epoch | 0.000384 s | 0.001736 s |
| One-time cache construction | 0.061236 s | 0.056149 s |
| Dense cache size | 59.35 MiB | 60.51 MiB |
| Original three-epoch run | 23.105945 s | 18.629345 s |
| Cached three-epoch run | 1.993544 s | 3.003544 s |
| Short-run total speedup | 11.59× | 6.20× |
| Original and cached final RMSE | 1.279154142 | 1.301556760 |

Raw timing samples, metrics, batch/state fingerprints, source/input hashes and
environment details are in [benchmark.json](benchmark.json). Sub-millisecond
preparation results measure cache slicing/gathering alone; use the complete-run
timings to assess the observed training benefit. These shortened experiments
use the three-epoch schedule and are not comparable to F12's 100-epoch result.

## Limits

Dense caches require 40 bytes per profile/visible-ID pair, plus temporary
construction arrays, model state and current batches. Larger sparse datasets
may need a sparse cache that expands only each batch. CPU timings are specific
to this host; accelerator throughput and full 100-epoch parity were not tested.
The existing set-based dataset ID order still requires a fixed `PYTHONHASHSEED`
for cross-process reproduction. Existing evaluation skips, test-only-profile
training behavior and item-based batch size 10 remain; runner consolidation
and configurable item-based batch size belong to F14.
