"""Batch parity against the pre-F13 preparation, including legacy edge cases."""

from collections import defaultdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cfrbm import item_based, user_based
from cfrbm.preparation import PreparedProfiles
from cfrbm.rbm import CFRBM
from cfrbm.utils import chunker, expand
from tests.test_rbm import STATE_NAMES


class LegacyPreparation:
    """Independent old nested loops, intentionally retaining list.index."""

    def __init__(self, profiles, visible_ids, tests):
        self.profiles = defaultdict(list, {key: list(value)
                                          for key, value in profiles.items()})
        self.visible_ids = visible_ids
        self.test_ids = list(tests)
        self.visible_index = {key: visible_ids.index(key) for key in visible_ids}

    def batch(self, ids):
        values, masks = [], []
        for profile_id in ids:
            ratings = [0.] * len(self.visible_ids)
            mask = [0] * (len(self.visible_ids) * 5)
            for visible_id, rating in self.profiles[profile_id]:
                ratings[self.visible_ids.index(visible_id)] = rating
                for offset in range(5):
                    mask[5 * self.visible_ids.index(visible_id) + offset] = 1
            values.append(expand(np.array([ratings])).astype('float32'))
            masks.append(mask)
        shape = (len(ids), len(self.visible_ids) * 5)
        return (np.array(values).reshape(shape),
                np.array(masks).reshape(shape).astype('float32'))

    def training_batches(self, epoch, batch_size):
        for ids in chunker(list(self.profiles), batch_size):
            yield self.batch(ids)

    def test_batches(self, batch_size):
        for ids in chunker(self.test_ids, batch_size):
            yield ids, self.batch(ids)[0]


class PreparationTest(unittest.TestCase):
    def test_batches_preserve_order_duplicates_rounding_and_cold_profiles(self):
        profiles = {'z': [('b', 5), ('a', 1), ('b', 2), ('c', 0)],
                    'a': [('c', 2.5), ('a', 3.5)], 'empty': []}
        tests = {'cold': [], 'a': [('b', 3)], 'z': [('missing', 1)]}
        visible = ['c', 'b', 'a', 'unused', 'b']
        for batch_size in (1, 2, 3, 10):
            old = LegacyPreparation(profiles, visible, tests)
            new = PreparedProfiles(profiles, visible, tests)
            for epoch in range(3):
                a = list(old.training_batches(epoch, batch_size))
                b = list(new.training_batches(epoch, batch_size))
                self.assertEqual(len(a), len(b))
                for old_pair, new_pair in zip(a, b):
                    for expected, actual in zip(old_pair, new_pair):
                        np.testing.assert_array_equal(actual, expected, strict=True)
                        self.assertTrue(actual.flags.c_contiguous)
                a, b = list(old.test_batches(batch_size)), list(new.test_batches(batch_size))
                self.assertEqual(len(a), len(b))
                for (old_ids, expected), (new_ids, actual) in zip(a, b):
                    self.assertEqual(old_ids, new_ids)
                    np.testing.assert_array_equal(actual, expected, strict=True)
        self.assertEqual(list(profiles), ['z', 'a', 'empty'])

    def test_cache_is_reused_and_does_not_consume_randomness(self):
        np.random.seed(17)
        expected = np.random.random(4)
        np.random.seed(17)
        profiles = {'p': [('v', 4)]}
        prepared = PreparedProfiles(profiles, ['v'], {})
        profiles['p'][0] = ('v', 1)
        for epoch in range(3):
            values, masks = next(prepared.training_batches(epoch, 10))
            self.assertTrue(np.shares_memory(values, prepared.values))
            self.assertTrue(np.shares_memory(masks, prepared.masks))
            np.testing.assert_array_equal(values, [[0, 0, 0, 1, 0]])
        np.testing.assert_array_equal(np.random.random(4), expected)
        self.assertEqual(list(prepared.test_batches(10)), [])
        empty = PreparedProfiles({}, [], {})
        self.assertEqual(list(empty.training_batches(0, 10)), [])
        self.assertEqual(empty.values.shape, (0, 0))

    def test_both_runners_replay_identical_seeded_training(self):
        config = dict(name='parity', number_hidden=3, epochs=3, ks=[1, 2, 3],
                      momentums=[0., .3, .7], l_w=[.001], l_v=[.002],
                      l_h=[.003], decay=.01, batch_size=2)
        users = [f'u{i}' for i in range(12)][::-1]
        movies = [f'm{i}' for i in range(12)][::-1]
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / 'train.tsv'
            dataset.write_text(''.join(f'u{i}\tm{i}\t{i % 5 + 1}\t0\n'
                                       for i in range(12)) + 'u0\tm0\t2\t0\n')
            config['name'] = str(Path(directory) / 'parity')
            for module, visible, profile in ((user_based, movies, 'u'),
                                              (item_based, users, 'm')):
                tests = {'cold': [(visible[0], 3)],
                         profile + '0': [(visible[-1], 2), ('unknown', 3),
                                         (visible[1], 4)]}
                outcomes = []
                for preparation in (LegacyPreparation, PreparedProfiles):
                    models, batches = [], []

                    class RecordingRBM(CFRBM):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, **kwargs)
                            models.append(self)

                        def cdk_fun(self, *args, **kwargs):
                            update = super().cdk_fun(*args, **kwargs)

                            def train(values, masks):
                                batches.append((values.clone(), masks.clone()))
                                return update(values, masks)
                            return train

                    np.random.seed(17)
                    with patch.object(module, 'PreparedProfiles', preparation), \
                            patch.object(module, 'CFRBM', RecordingRBM), \
                            redirect_stdout(io.StringIO()):
                        module.run('test', dataset, config, users, movies, tests,
                                   None, '\t', device='cpu')
                    result = json.loads(Path(config['name'] + '_test.json').read_text())
                    outcomes.append((models[0], batches, result))
                before, after = outcomes
                self.assertEqual(before[2], after[2])
                self.assertEqual(len(before[1]), len(after[1]))
                for old_pair, new_pair in zip(before[1], after[1]):
                    for old, new in zip(old_pair, new_pair):
                        torch.testing.assert_close(old, new, rtol=0, atol=0)
                for name in STATE_NAMES:
                    torch.testing.assert_close(getattr(before[0], name),
                                               getattr(after[0], name), rtol=0, atol=0)
                torch.testing.assert_close(before[0].generator.get_state(),
                                           after[0].generator.get_state(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
