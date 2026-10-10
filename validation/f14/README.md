# F14 runner consolidation parity

`parity.json` records three seeded CPU epochs on MovieLens u1 for both
orientations at batch sizes 10 and 7. All four comparisons have identical
training/evaluation batch hashes, per-epoch metrics, six model/velocity tensor
hashes and final sampling RNG state.

The baseline is commit `9e8fa056926bdf7087b8bc92bdf5da3b181527bf`, immediately
before F14. Batch size 10 compares both original runners without modifications.
User mode also matches the original at batch size 7. For item mode at size 7,
the baseline's two `batch_size = 10` assignments are replaced with configuration
lookups; this checks that honoring the configured size is the only behavioral
change. It does not claim unchanged results versus the old hardcoded size 10.

Reproduce from the repository root with the MovieLens dataset extracted:

```sh
PYTHONHASHSEED=17 PYTHONPATH=. .venv/bin/python validation/verify_runner.py > /tmp/f14-parity.json
make test PYTHON=.venv/bin/python
```

The evidence includes input and application-source SHA-256 fingerprints,
Python/NumPy/PyTorch versions, configuration and metrics. NumPy initialization
and model sampling use seed 17; PyTorch uses one CPU thread. Fixed hash seed
preserves dataset ID ordering across processes.

The full suite ran 42 tests: 40 passed, with CUDA and MPS unavailable in the
sandbox. It covers both shared CLI orientations and legacy entry points;
configured sizes 1, 4, 10 and 20; short final batches; asymmetric user/item counts;
test-only profiles; schedules; experiment/config dispatch; and invalid orientation.

Parity evidence covers three CPU epochs, not full 100-epoch or accelerator runs.
Existing result filenames and evaluation behavior are preserved, including
rounded predictions and unknown-visible-ID skips. Running both orientations
with the same experiment/config names overwrites the same output file.
