"""F5 parity contract for the F6 PyTorch CFRBM, using the F4 NumPy oracle.

Run with ``python -m unittest tests.test_rbm -v`` (also in ``make test``).
Only the legacy Theano source is skipped. Once it is replaced, missing torch,
import errors and API mismatches fail normally rather than hiding regressions.

TorchAdapter is the only backend-specific boundary. It assumes the existing
state names, constructor and cdk_fun(vis, masks, **options) -> callable API.
If F6 changes that factory signature, adapt train() here. Sampling must call
torch.bernoulli; we replace that operation, not the model's sampling methods,
so normalization and the complete CD path are still exercised. No RNG seed or
random-number sequence is part of the contract. These tests preserve F10–F12
quirks until those features deliberately update the oracle and expectations.
"""

import ast
from contextlib import contextmanager
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from tests import rbm_reference as reference


STATE_NAMES = ('weights', 'vbias', 'hbias', 'prev_gw', 'prev_gv', 'prev_gh')
RTOL = 1e-5
ATOL = 1e-7


class TorchAdapter:
    def __init__(self, state):
        import torch
        from cfrbm.rbm import CFRBM

        self.torch = torch
        self.model = CFRBM(*state[0].shape, device="cpu")
        with torch.no_grad():
            for name, value in zip(STATE_NAMES, state):
                target = getattr(self.model, name)
                target.copy_(torch.as_tensor(value, dtype=target.dtype,
                                             device=target.device))

    def tensor(self, value):
        return self.torch.as_tensor(value, dtype=self.model.weights.dtype,
                                    device=self.model.weights.device)

    @staticmethod
    def numpy(value):
        return value.detach().cpu().numpy().copy()

    def state(self):
        return tuple(self.numpy(getattr(self.model, name)) for name in STATE_NAMES)

    def prop_up(self, visible):
        return self.numpy(self.model.prop_up(self.tensor(visible)))

    def prop_down(self, hidden):
        return self.numpy(self.model.prop_down(self.tensor(hidden)))

    @contextmanager
    def fixed_samples(self, samples):
        queue = iter(samples)

        def bernoulli(probabilities, *args, **kwargs):
            try:
                sample = next(queue)
            except StopIteration:
                raise AssertionError('CFRBM requested an extra binary sample') from None
            if tuple(probabilities.shape) != sample.shape:
                raise AssertionError('CFRBM requested a sample with the wrong shape')
            result = self.torch.as_tensor(sample, dtype=probabilities.dtype,
                                          device=probabilities.device).clone()
            if kwargs.get('out') is not None:
                kwargs['out'].copy_(result)
                return kwargs['out']
            return result

        with patch.object(self.torch, 'bernoulli', side_effect=bernoulli):
            yield
        if next(queue, None) is not None:
            raise AssertionError('CFRBM did not consume all fixed binary samples')

    def sample_hidden(self, visible):
        return tuple(self.numpy(x) for x in self.model.sample_hidden(self.tensor(visible)))

    def sample_visible(self, hidden):
        return tuple(self.numpy(x) for x in self.model.sample_visible(self.tensor(hidden)))

    def gradient(self, *values):
        return tuple(self.numpy(x) for x in
                     self.model.gradient(*(self.tensor(v) for v in values)))

    def train(self, visible, masks, **options):
        visible, masks = self.tensor(visible), self.tensor(masks)
        update = self.model.cdk_fun(visible, masks, **options)
        if not callable(update):
            raise AssertionError('cdk_fun must return a training callable')
        return lambda: update(visible, masks)


class RBMParityChecks:
    """Reusable checks: supply an adapter_factory accepting the six state arrays."""

    def setUp(self):
        # Three users, two rating groups and three hidden units; no random data.
        self.initial = (
            np.linspace(-.65, .8, 30, dtype=np.float32).reshape(10, 3),
            np.linspace(-.4, .5, 10, dtype=np.float32),
            np.array([.2, -.3, .1], dtype=np.float32),
            np.linspace(.12, -.17, 30, dtype=np.float32).reshape(10, 3),
            np.linspace(-.2, .25, 10, dtype=np.float32),
            np.array([-.15, .2, -.1], dtype=np.float32),
        )
        self.visible = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.float32)
        self.masks = np.array([[1] * 10, [1] * 5 + [0] * 5, [0] * 10],
                              dtype=np.float32)
        self.hidden = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        # Independent Bernoulli draws deliberately include empty and multi-hot groups.
        self.negative = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
        ], dtype=np.float32)
        self.last_hidden = 1 - self.hidden
        self.backend = self.adapter_factory(tuple(a.copy() for a in self.initial))

    def assert_close(self, actual, expected):
        # allclose broadcasts, which would otherwise conceal shape regressions.
        self.assertEqual(actual.shape, expected.shape)
        assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=False)

    def test_prop_up(self):
        self.assert_close(self.backend.prop_up(self.visible),
                          reference.prop_up(self.visible, self.initial[0], self.initial[2]))

    def test_prop_down(self):
        self.assert_close(self.backend.prop_down(self.hidden),
                          reference.prop_down(self.hidden, self.initial[0], self.initial[1]))

    def test_fixed_hidden_sample(self):
        with self.backend.fixed_samples([self.hidden]):
            sample, probabilities = self.backend.sample_hidden(self.visible)
        self.assert_close(sample, self.hidden)
        self.assert_close(probabilities,
                          reference.prop_up(self.visible, self.initial[0], self.initial[2]))

    def test_fixed_visible_sample_and_five_way_normalization(self):
        expected = reference.sample_visible(
            self.hidden, self.initial[0], self.initial[1], self.negative)
        with self.backend.fixed_samples([self.negative]):
            actual = self.backend.sample_visible(self.hidden)
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected):
            self.assert_close(value, target)
        self.assert_close(actual[1].reshape(3, 2, 5).sum(axis=2), np.ones((3, 2)))

    def test_masked_gradient(self):
        h2p = reference.prop_up(self.negative, self.initial[0], self.initial[2])
        values = (self.visible, self.hidden, self.negative, h2p, self.masks)
        actual = self.backend.gradient(*values)
        expected = reference.gradient(*values)
        self.assertEqual(len(actual), len(expected))
        for value, target in zip(actual, expected):
            self.assert_close(value, target)

    def check_updates(self, options, steps=1):
        expected = tuple(a.copy() for a in self.initial)
        samples = [self.hidden, self.negative, self.last_hidden] * steps
        with self.backend.fixed_samples(samples):
            update = self.backend.train(self.visible, self.masks, **options)
            # Constructing the callable must not perform a parameter update.
            for value, target in zip(self.backend.state(), expected):
                self.assert_close(value, target)
            for _ in range(steps):
                h2p = reference.prop_up(self.negative, expected[0], expected[2])
                gradients = reference.gradient(self.visible, self.hidden,
                                               self.negative, h2p, self.masks)
                expected = reference.parameter_update(*expected, gradients, **options)
                update()
                actual = self.backend.state()
                self.assertEqual(len(actual), len(expected))
                for name, value, target in zip(STATE_NAMES, actual, expected):
                    with self.subTest(state=name):
                        self.assertEqual(value.dtype, np.dtype('float32'))
                        self.assert_close(value, target)

    def test_update_defaults(self):
        self.check_updates({})

    def test_update_with_decay_and_raw_gradient_momentum(self):
        self.check_updates(dict(w_lr=.03, v_lr=.02, h_lr=.04, decay=.07, momentum=.6))

    def test_reused_callable_updates_gradient_memory(self):
        self.check_updates(dict(w_lr=.03, v_lr=.02, h_lr=.04, decay=.07, momentum=.6),
                           steps=2)


class TorchRBMParityTest(RBMParityChecks, unittest.TestCase):
    adapter_factory = TorchAdapter

    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / 'cfrbm' / 'rbm.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        # Inspect imports without importing the incompatible Theano package.
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            if any(name.split('.')[0] == 'theano' for name in modules):
                raise unittest.SkipTest('F6 pending: CFRBM still uses Theano; '
                                        'PyTorch parity has not been verified')


if __name__ == '__main__':
    unittest.main()
