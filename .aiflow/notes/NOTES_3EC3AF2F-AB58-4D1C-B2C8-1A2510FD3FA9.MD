# F13 completed

Both runners now cache profiles and masks once per run. The existing launch
commands work unchanged. Seeded MovieLens batches, three-epoch metrics, final
model state and RNG state match the original runners exactly.

## Performance and memory

Three-epoch CPU runs were 11.59× faster for user-based and 6.20× for item-based
training. Dense caches used 59.35 MiB and 60.51 MiB respectively on MovieLens u1.
Budget 40 bytes per profile/visible-ID pair plus temporary construction arrays,
model state and batches. Add sparse caching before using much larger datasets.

## Reproduce the checks

From the repository root with the existing `.venv` and MovieLens files:

```sh
make test PYTHON=.venv/bin/python
PYTHONHASHSEED=17 PYTHONPATH=. .venv/bin/python validation/benchmark_preparation.py > /tmp/f13-benchmark.json
```

See `validation/f13/README.md` and `benchmark.json` for evidence and protocol.
Keep `PYTHONHASHSEED=17` to reproduce the dataset's existing set-based ID order.
Accelerator throughput and full 100-epoch parity remain unverified. Existing
evaluation skips and test-only-profile insertion are preserved; changing those
requires a separate correctness change. F14 still owns runner consolidation
and configurable item-based batch size.
