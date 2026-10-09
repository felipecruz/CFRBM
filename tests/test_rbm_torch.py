"""Backend contracts beyond the fixed CD-1 parity harness."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import torch

from cfrbm.rbm import CFRBM
from tests import rbm_reference as reference
from tests.test_rbm import STATE_NAMES, TorchAdapter


class TorchRBMContractTest(unittest.TestCase):
    def test_initialization_preserves_numpy_normal_and_accepts_initial_bias(self):
        rng = np.random.get_state()
        try:
            np.random.seed(23)
            expected = np.random.normal(0, .1, (10, 3)).astype(np.float32)
            np.random.seed(23)
            bias = np.arange(10, dtype=np.float32)
            model = CFRBM(10, 3, initial_v=bias, debug=True)
        finally:
            np.random.set_state(rng)
        assert_array_equal(model.weights.numpy(), expected)
        assert_array_equal(model.vbias.numpy(), bias)
        bias[:] = -1
        self.assertEqual(model.vbias[-1].item(), 9)
        for name in ('hbias', 'prev_gw', 'prev_gv', 'prev_gh'):
            self.assertEqual(torch.count_nonzero(getattr(model, name)).item(), 0)
        for name in STATE_NAMES:
            self.assertEqual(getattr(model, name).dtype, torch.float32)

    def test_npy_loading_preserves_suffixes_and_overrides_initial_bias(self):
        weights = np.arange(30, dtype=np.float32).reshape(10, 3) / 100
        vbias = np.linspace(-1, 1, 10, dtype=np.float32)
        hbias = np.array([.1, -.2, .3], dtype=np.float32)
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / 'checkpoint'
            for suffix, value in [('W', weights), ('b', vbias), ('h', hbias)]:
                np.save(f'{prefix}.{suffix}.npy', value)
            model = CFRBM(10, 3, initial_v=[99] * 10, initial_weigths=prefix)
        for name, expected in [('weights', weights), ('vbias', vbias), ('hbias', hbias)]:
            assert_array_equal(getattr(model, name).numpy(), expected)

    def test_seeded_sampling_is_local_reproducible_and_advances(self):
        first, second = CFRBM(10, 3), CFRBM(10, 3)
        second.weights.copy_(first.weights)
        visible = torch.zeros(100, 10)
        global_state = torch.random.get_rng_state().clone()
        expected_rng = torch.Generator().manual_seed(17)
        expected = torch.bernoulli(first.prop_up(visible), generator=expected_rng)
        sample, _ = first.sample_hidden(visible)
        torch.testing.assert_close(sample, expected, rtol=0, atol=0)
        torch.testing.assert_close(sample, second.sample_hidden(visible)[0], rtol=0, atol=0)
        next_sample = first.sample_hidden(visible)[0]
        self.assertFalse(torch.equal(sample, next_sample))
        torch.testing.assert_close(next_sample, second.sample_hidden(visible)[0], rtol=0, atol=0)
        hidden = torch.ones(100, 3)
        torch.testing.assert_close(first.sample_visible(hidden)[0],
                                   second.sample_visible(hidden)[0], rtol=0, atol=0)
        torch.testing.assert_close(torch.random.get_rng_state(), global_state, rtol=0, atol=0)

    def test_prediction_factory_is_lazy_reusable_and_returns_probabilities(self):
        model = CFRBM(10, 3)
        batch = torch.zeros(2, 10, requires_grad=True)
        hidden = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        expected = reference.sample_visible(hidden.numpy(), model.weights.numpy(),
                                            model.vbias.numpy(), np.zeros((2, 10)))[1]
        with patch.object(torch, 'bernoulli', side_effect=[hidden, torch.zeros(2, 10)] * 2) as draw:
            predict = model.predict()
            compatible_predict = model.predict(batch)
            draw.assert_not_called()
            for factory in (predict, compatible_predict):
                output = factory(batch)
                assert_allclose(output.numpy(), expected, rtol=1e-5, atol=1e-7)
                self.assertFalse(output.requires_grad)

    def test_cd2_uses_final_transition_and_updates_in_place_without_autograd(self):
        state = (np.linspace(-.2, .3, 30, dtype=np.float32).reshape(10, 3),
                 np.zeros(10, dtype=np.float32), np.zeros(3, dtype=np.float32),
                 np.full((10, 3), .1, dtype=np.float32),
                 np.full(10, -.1, dtype=np.float32), np.full(3, .2, dtype=np.float32))
        backend = TorchAdapter(state)
        model = backend.model
        identities = [(getattr(model, name), getattr(model, name).data_ptr())
                      for name in STATE_NAMES]
        v0 = np.zeros((2, 10), dtype=np.float32)
        v1 = np.array([[1, 0, 1, 0, 0] * 2, [0, 1, 0, 0, 0] * 2], dtype=np.float32)
        v2 = 1 - v1
        h0 = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
        h1 = 1 - h0
        masks = np.array([[1] * 10, [1] * 5 + [0] * 5], dtype=np.float32)
        options = dict(w_lr=.03, v_lr=.02, h_lr=.04, decay=.07, momentum=.6)
        gradients = reference.gradient(v1, h1, v2, reference.prop_up(v2, state[0], state[2]), masks)
        expected = reference.parameter_update(*state, gradients, **options)
        batch = backend.tensor(v0).requires_grad_()
        with backend.fixed_samples([h0, v1, h0, h1, v2, h0]):
            update = model.cdk_fun(None, None, k=2, **options)
            update(batch, backend.tensor(masks))
        for name, (original, pointer), actual, target in zip(
                STATE_NAMES, identities, backend.state(), expected):
            self.assertIs(getattr(model, name), original)
            self.assertEqual(original.data_ptr(), pointer)
            self.assertFalse(original.requires_grad)
            self.assertIsNone(original.grad_fn)
            assert_allclose(actual, target, rtol=1e-5, atol=1e-7)
        self.assertIsNone(batch.grad)


if __name__ == '__main__':
    unittest.main()
