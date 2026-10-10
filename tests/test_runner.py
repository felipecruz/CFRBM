"""Orientation, configured batching, compatibility and experiment dispatch."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cfrbm import item_based, runner, user_based
from cfrbm.device import parse_args
from cfrbm.rbm import CFRBM
from tests.test_rbm import STATE_NAMES


class RunnerTest(unittest.TestCase):
    def test_orientations_batches_and_compatibility(self):
        users = [f'u{i}' for i in range(13)]
        movies = [f'm{i}' for i in range(11)]
        entries = [(user, movies[i % 11], i % 5 + 1)
                   for i, user in enumerate(users)]
        with TemporaryDirectory() as directory:
            dataset = Path(directory) / 'train.tsv'
            dataset.write_text(''.join(f'{u}\t{m}\t{r}\t0\n' for u, m, r in entries))
            config = dict(name=str(Path(directory) / 'result'), number_hidden=3,
                          epochs=3, ks=[1, 2, 3], momentums=[0., .3, .7],
                          l_w=[.001], l_v=[.002], l_h=[.003], decay=.01)
            for orientation, wrapper in [('user', user_based), ('item', item_based)]:
                profiles, visible = (users, movies) if orientation == 'user' else (movies, users)
                tests = {pid: [(visible[0], 3)] for pid in profiles}
                # Test-only profiles join training after epoch zero, as before.
                tests['cold'] = [(visible[1], 2)]
                for size in (1, 4, 10, 20):
                    with self.subTest(orientation=orientation, batch_size=size):
                        config['batch_size'] = size
                        outcomes = []
                        for call, options in ((runner.run, {'orientation': orientation}),
                                              (wrapper.run, {})):
                            models, training, evaluation = [], [], []

                            class RecordingRBM(CFRBM):
                                def __init__(self, *args, **kwargs):
                                    super().__init__(*args, **kwargs)
                                    models.append(self)

                                def cdk_fun(self, *args, **kwargs):
                                    update = super().cdk_fun(*args, **kwargs)

                                    def train(values, masks):
                                        training.append(values.clone())
                                        return update(values, masks)
                                    return train

                                def predict(self):
                                    predict = super().predict()

                                    def predict_batch(values):
                                        evaluation.append(len(values))
                                        return predict(values)
                                    return predict_batch

                            np.random.seed(17)
                            with patch.object(runner, 'CFRBM', RecordingRBM), redirect_stdout(io.StringIO()):
                                call('test', dataset, config, users, movies, tests,
                                     None, '\t', device='cpu', **options)
                            metrics = json.loads(Path(config['name'] + '_test.json').read_text())
                            outcomes.append((models[0], metrics))

                            def sizes(count):
                                return [min(size, count - i) for i in range(0, count, size)]

                            expected = sizes(len(profiles)) + sizes(len(profiles) + 1) * 2
                            self.assertEqual([len(batch) for batch in training], expected)
                            self.assertEqual(evaluation, sizes(len(tests)) * 3)
                            first_epoch = torch.cat(training[:len(sizes(len(profiles)))])
                            expected_values = torch.zeros(len(profiles), len(visible) * 5)
                            for user, movie, rating in entries:
                                pid, vid = (user, movie) if orientation == 'user' else (movie, user)
                                expected_values[profiles.index(pid), visible.index(vid) * 5 + rating - 1] = 1
                            torch.testing.assert_close(first_epoch, expected_values, rtol=0, atol=0)
                        self.assertEqual(outcomes[0][1], outcomes[1][1])
                        for name in STATE_NAMES:
                            torch.testing.assert_close(getattr(outcomes[0][0], name),
                                                       getattr(outcomes[1][0], name), rtol=0, atol=0)
                        self.assertTrue(torch.equal(outcomes[0][0].generator.get_state(),
                                                    outcomes[1][0].generator.get_state()))

    def test_dispatches_all_experiments_and_configs_in_selected_orientation(self):
        experiments = [dict(name=f'e{i}', train_path=f'train{i}', test_path=f'test{i}',
                            sep='\t', configs=[{'name': 'a'}, {'name': 'b'}]) for i in range(2)]
        for orientation in ('user', 'item'):
            with self.subTest(orientation=orientation), \
                    patch.object(runner, 'read_experiment', return_value=experiments), \
                    patch.object(runner, 'load_dataset', return_value=(['u'], ['m'], {})) as load, \
                    patch.object(runner, 'run') as run:
                runner.main(['experiment.json', '--device', 'cpu', '--orientation', orientation])
                self.assertEqual(load.call_count, 2)
                self.assertEqual(run.call_count, 4)
                self.assertEqual([call.args[0] for call in run.call_args_list], ['e0', 'e0', 'e1', 'e1'])
                self.assertEqual([call.args[2]['name'] for call in run.call_args_list], ['a', 'b'] * 2)
                for call in load.call_args_list:
                    self.assertEqual(call.kwargs['user_based'], orientation == 'user')
                for call in run.call_args_list:
                    self.assertEqual(call.kwargs, dict(device=torch.device('cpu'), orientation=orientation))

    def test_orientation_defaults_and_invalid_values(self):
        argv = ['experiment.json', '--device', 'cpu']
        self.assertEqual(parse_args(argv).orientation, 'user')
        self.assertEqual(parse_args(argv, orientation='item').orientation, 'item')
        self.assertEqual(parse_args(['--orientation', 'item', *argv]).orientation, 'item')
        with redirect_stdout(io.StringIO()), patch('sys.stderr', new=io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            parse_args([*argv, '--orientation', 'invalid'])
        self.assertEqual(error.exception.code, 2)
        with self.assertRaisesRegex(ValueError, 'orientation'):
            runner.run('test', 'missing', {}, [], [], {}, None, '\t', orientation='invalid')


if __name__ == '__main__':
    unittest.main()
