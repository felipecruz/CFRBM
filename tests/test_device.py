"""Device policy and real training coverage on CPU and available accelerators."""

import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cfrbm.device import parse_args, select_device
from cfrbm.rbm import CFRBM
from tests.test_rbm import STATE_NAMES


class DeviceSelectionTest(unittest.TestCase):
    def test_auto_priority_and_explicit_cpu(self):
        for cuda, mps, expected in [(True, True, 'cuda:1'),
                                     (True, False, 'cuda:1'),
                                     (False, True, 'mps'),
                                     (False, False, 'cpu')]:
            with self.subTest(cuda=cuda, mps=mps), \
                    patch('torch.cuda.is_available', return_value=cuda), \
                    patch('torch.cuda.current_device', return_value=1), \
                    patch('torch.cuda.device_count', return_value=2), \
                    patch('torch.backends.mps.is_available', return_value=mps):
                self.assertEqual(str(select_device()), expected)
                self.assertEqual(str(select_device('auto')), expected)
                self.assertEqual(select_device('cpu'), torch.device('cpu'))
                self.assertEqual(select_device(torch.device('cpu')), torch.device('cpu'))
                if cuda:
                    self.assertEqual(str(select_device('cuda:0')), 'cuda:0')
                if mps:
                    self.assertEqual(str(select_device('mps')), 'mps')

    def test_invalid_and_unavailable_devices_fail(self):
        with patch('torch.cuda.is_available', return_value=False), \
                patch('torch.backends.mps.is_available', return_value=False):
            for device in ('cuda', 'cuda:0', 'mps', 'bogus', 'meta', 'cpu:2', 'mps:1'):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    select_device(device)
        with patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.device_count', return_value=2):
            with self.assertRaisesRegex(ValueError, 'index 2'):
                select_device('cuda:2')

    def test_cli_option_can_precede_or_follow_experiment(self):
        for argv in (['example.json', '--device', 'cpu'],
                     ['--device', 'cpu', 'example.json']):
            args = parse_args(argv)
            self.assertEqual(args.experiment, 'example.json')
            self.assertEqual(args.device, torch.device('cpu'))


class DeviceTrainingTest(unittest.TestCase):
    def check_training(self, device):
        # Exercise .npy loading as well as all six parameter/momentum tensors.
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / 'weights'
            for suffix, values in [('W', np.full((10, 3), .1)),
                                   ('b', np.zeros(10)), ('h', np.zeros(3))]:
                np.save(f'{prefix}.{suffix}.npy', values)
            model = CFRBM(10, 3, initial_weigths=prefix, device=device)
            twin = CFRBM(10, 3, initial_weigths=prefix, device=device)
        visible = torch.zeros(64, 10, device=model.device)
        visible[:, 0] = 1
        visible[:, 7] = 1
        masks = torch.ones_like(visible)
        masks[:, 5:] = 0
        before = model.weights.clone()
        rng = model.generator.get_state().clone()
        for current in (model, twin):
            update = current.cdk_fun(None, None, k=3, momentum=.3, decay=.01)
            update(visible, masks)
            update(visible, masks)
        self.assertFalse(torch.equal(rng, model.generator.get_state()))
        self.assertFalse(torch.equal(before, model.weights))
        for name in STATE_NAMES:
            value = getattr(model, name)
            self.assertEqual(value.device.type, model.device.type)
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value).all().item())
            self.assertFalse(value.requires_grad)
            torch.testing.assert_close(value, getattr(twin, name), rtol=0, atol=0)
        probabilities = model.predict()(visible)
        self.assertEqual(probabilities.device.type, model.device.type)
        self.assertTrue(torch.isfinite(probabilities).all().item())
        torch.testing.assert_close(probabilities.reshape(-1, 5).sum(1),
                                   torch.ones(128, device=model.device))

    def test_cpu_training(self):
        self.check_training('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_training(self):
        self.check_training('mps')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_training(self):
        self.check_training('cuda')

    def test_both_clis_train_and_write_results(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root))
        devices = ['cpu', 'auto']
        devices += [d for d, available in [('mps', torch.backends.mps.is_available()),
                                           ('cuda', torch.cuda.is_available())] if available]
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            train = directory / 'train.tsv'
            test = directory / 'test.tsv'
            train.write_text('1\t1\t5\t0\n1\t2\t3\t0\n2\t1\t4\t0\n2\t2\t2\t0\n')
            test.write_text('1\t2\t3\t0\n2\t1\t4\t0\n')
            config = dict(name='smoke', number_hidden=3, epochs=2, ks=[1, 3],
                          momentums=[0., .3], l_w=[.001], l_v=[.001],
                          l_h=[.001], decay=.01, batch_size=1)
            experiment = directory / 'experiment.json'
            experiment.write_text(json.dumps({'experiments': [dict(
                name='device', train_path=str(train), test_path=str(test),
                sep='\t', configs=[config])]}))
            result = directory / 'smoke_device.json'
            for module in ('cfrbm.user_based', 'cfrbm.item_based'):
                for device in devices:
                    with self.subTest(module=module, device=device):
                        command = [sys.executable, '-m', module, str(experiment)]
                        if device != 'auto':  # Also test the default CLI selection.
                            command += ['--device', device]
                        completed = subprocess.run(command, cwd=directory, env=env,
                                                   capture_output=True, text=True, timeout=60)
                        self.assertEqual(completed.returncode, 0, completed.stderr)
                        self.assertIn(f'Using device: {select_device(device)}', completed.stdout)
                        metrics = json.loads(result.read_text())['results']
                        self.assertEqual(len(metrics), 2)
                        self.assertTrue(all(np.isfinite(row[key]) for row in metrics
                                            for key in ('mae', 'rmse')))
                        result.unlink()
                failed = subprocess.run([sys.executable, '-m', module, str(experiment),
                                         '--device', 'invalid'], cwd=directory, env=env,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(failed.returncode, 2)
                self.assertIn('Invalid device', failed.stderr)
                self.assertFalse(result.exists())


if __name__ == '__main__':
    unittest.main()
