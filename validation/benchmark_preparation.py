"""Reproduce F13 batch parity, preparation timings and short training parity.

Run from the repository root with PYTHONHASHSEED=17 and PYTHONPATH=.
Writes JSON to stdout; baseline sources are read from the specified Git commit.
"""

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
from tempfile import TemporaryDirectory
import time
import types

import numpy as np
import torch

from cfrbm import item_based, user_based
from cfrbm.dataset import load_dataset
from cfrbm.preparation import PreparedProfiles
from cfrbm.rbm import CFRBM
from tests.test_preparation import LegacyPreparation
from tests.test_rbm import STATE_NAMES


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def consume(prepared, epoch):
    for values, masks in prepared.training_batches(epoch, 10):
        pass
    for ids, values in prepared.test_batches(10):
        pass


def compare_batches(old, new):
    for epoch in range(2):
        old_batches = list(old.training_batches(epoch, 10))
        new_batches = list(new.training_batches(epoch, 10))
        assert len(old_batches) == len(new_batches)
        for before, after in zip(old_batches, new_batches):
            for a, b in zip(before, after):
                np.testing.assert_array_equal(a, b, strict=True)
        old_tests, new_tests = list(old.test_batches(10)), list(new.test_batches(10))
        assert len(old_tests) == len(new_tests)
        for (a_ids, a), (b_ids, b) in zip(old_tests, new_tests):
            assert a_ids == b_ids
            np.testing.assert_array_equal(a, b, strict=True)


def runner_trial(module, users, movies, tests, config):
    models = []
    batches_hash = hashlib.sha256()

    class RecordingRBM(CFRBM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            models.append(self)

        def cdk_fun(self, *args, **kwargs):
            update = super().cdk_fun(*args, **kwargs)

            def train(values, masks):
                for value in (values, masks):
                    batches_hash.update(str(tuple(value.shape)).encode())
                    batches_hash.update(value.numpy().tobytes())
                update(values, masks)
            return train

        def predict(self):
            predict = super().predict()

            def predict_batch(values):
                batches_hash.update(str(tuple(values.shape)).encode())
                batches_hash.update(values.numpy().tobytes())
                return predict(values)
            return predict_batch

    original = module.CFRBM
    module.CFRBM = RecordingRBM
    try:
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            config = dict(config, name=str(Path(directory) / 'result'))
            np.random.seed(17)
            started = time.perf_counter()
            module.run('parity', ROOT / 'ml-100k/u1.base', config, users, movies,
                       tests, None, '\t', device='cpu')
            elapsed = time.perf_counter() - started
            metrics = json.loads(Path(config['name'] + '_parity.json').read_text())['results']
    finally:
        module.CFRBM = original
    model = models[0]
    state = {name: hashlib.sha256(getattr(model, name).numpy().tobytes()).hexdigest()
             for name in STATE_NAMES}
    state['rng'] = hashlib.sha256(model.generator.get_state().numpy().tobytes()).hexdigest()
    return dict(seconds=elapsed, metrics=metrics, state_sha256=state,
                batches_sha256=batches_hash.hexdigest())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='34a0ce7b2419fca20fef771afe05232506603b3f')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((ROOT / 'ubased.json').read_text())['experiments'][0]['configs'][0]
    config = dict(config, epochs=3)
    results = dict(baseline_commit=args.baseline, python=platform.python_version(),
                   platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                   seed=17, device='cpu', torch_threads=1, training_epochs=3,
                   repeats=args.repeats,
                   input_sha256={p: digest(ROOT / p) for p in
                                 ('ml-100k/u1.base', 'ml-100k/u1.test', 'ubased.json')},
                   source_sha256={str(p.relative_to(ROOT)): digest(p)
                                  for p in sorted((ROOT / 'cfrbm').glob('*.py'))},
                   orientations={})
    for mode, module in (('user', user_based), ('item', item_based)):
        users, movies, tests = load_dataset(ROOT / 'ml-100k/u1.base',
                                           ROOT / 'ml-100k/u1.test', '\t', mode == 'user')
        visible = movies if mode == 'user' else users
        profiles = defaultdict(list)
        with (ROOT / 'ml-100k/u1.base').open() as stream:
            for line in stream:
                user, movie, rating, _ = line.strip().split('\t')
                pid, vid = (user, movie) if mode == 'user' else (movie, user)
                profiles[pid].append((vid, float(rating)))
        old = LegacyPreparation(profiles, visible, tests)
        start = time.perf_counter()
        new = PreparedProfiles(profiles, visible, tests)
        setup_seconds = time.perf_counter() - start
        compare_batches(old, new)
        timings = {'legacy': [], 'cached': []}
        # Alternate timing order; parity comparison above warms both paths.
        for repeat in range(args.repeats):
            order = [('legacy', old), ('cached', new)]
            if repeat % 2:
                order.reverse()
            for name, prepared in order:
                start = time.perf_counter()
                consume(prepared, 1)
                timings[name].append(time.perf_counter() - start)
        medians = {name: statistics.median(values) for name, values in timings.items()}
        assert medians['cached'] < medians['legacy'], timings
        source = subprocess.check_output(
            ['git', 'show', f'{args.baseline}:cfrbm/{mode}_based.py'], cwd=ROOT, text=True)
        baseline = types.ModuleType(f'cfrbm._baseline_{mode}')
        baseline.__package__ = 'cfrbm'
        exec(compile(source, f'<baseline-{mode}>', 'exec'), baseline.__dict__)
        before = runner_trial(baseline, users, movies, tests, config)
        after = runner_trial(module, users, movies, tests, config)
        for key in ('metrics', 'state_sha256', 'batches_sha256'):
            assert before[key] == after[key], (mode, key)
        results['orientations'][mode] = dict(
            exact_batch_parity=True, exact_training_parity=True,
            profile_count=len(new.values), visible_count=len(visible),
            cache_bytes=new.values.nbytes + new.masks.nbytes,
            setup_seconds=setup_seconds, preparation_seconds=timings,
            median_preparation_seconds=medians,
            preparation_speedup=medians['legacy'] / medians['cached'],
            baseline_training=before, cached_training=after,
            training_speedup=before['seconds'] / after['seconds'])
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
