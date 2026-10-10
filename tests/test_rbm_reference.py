"""Hand-calculated checks of the NumPy oracle; backend parity belongs to F5."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from tests import rbm_reference as reference


class RBMReferenceTest(unittest.TestCase):
    def test_propagation_in_both_directions(self):
        # Integer multiples of log(3) give exact rational sigmoid targets.
        weights = np.log(3) * np.array([[1., -1.], [0., 1.], [-1., 0.]])
        hbias = np.log(3) * np.array([1., -1.])
        vbias = np.log(3) * np.array([0., 1., -1.])
        visible = np.array([[1., 0., 1.], [0., 1., 0.]])
        hidden = np.array([[1., 0.], [1., 1.]])

        assert_allclose(reference.prop_up(visible, weights, hbias),
                        [[.75, .1], [.75, .5]])
        assert_allclose(reference.prop_down(hidden, weights, vbias, k=3),
                        [[27/55, 27/55, 1/55], [9/91, 81/91, 1/91]])

    def test_visible_softmax_and_fixed_one_hot_samples(self):
        # Two users and two rating groups catch normalization across wrong axes.
        probabilities = np.array([.1, .2, .3, .4, .5, .8, .6, .4, .2, .1])
        vbias = np.log(probabilities)
        weights = np.zeros((10, 2))
        hidden = np.array([[1., 0.], [0., 1.]])
        samples = np.array([[1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0]])
        expected = [[1/15, 2/15, 3/15, 4/15, 5/15,
                     8/21, 6/21, 4/21, 2/21, 1/21]] * 2

        sample, normalized = reference.sample_visible(
            hidden, weights, vbias, samples)
        assert_array_equal(sample, samples)
        assert_allclose(normalized, expected)
        assert_allclose(normalized.reshape(2, 2, 5).sum(axis=2), 1)

    def test_gradient_masks_observed_ratings_and_keeps_unmasked_hidden_bias(self):
        v1 = np.array([[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]])
        h1 = np.array([[1., 0.], [0., 1.]])
        v2 = np.array([[0., 1., 1., 0., 0.], [1., 0., 0., 1., 0.]])
        h2p = np.array([[.25, .75], [.6, .4]])
        masks = np.array([[1., 1., 1., 1., 1.], [0., 0., 0., 0., 0.]])

        gw, gv, gh = reference.gradient(v1, h1, v2, h2p, masks)
        assert_allclose(gw, [[.5, 0], [-.125, -.375], [-.125, -.375], [0, 0], [0, 0]])
        assert_allclose(gv, [.5, -.5, -.5, 0, 0])
        assert_allclose(gh, [.075, -.075])

    def test_update_decay_velocity_momentum_and_float32_cast(self):
        weights = np.array([[1., -2.], [3., -4.]])
        vbias = np.array([.5, -.5])
        hbias = np.array([1., -1.])
        prev_gw = np.array([[.2, -.4], [.6, -.8]])
        prev_gv = np.array([.4, -.2])
        prev_gh = np.array([-.6, .8])
        gradients = (np.array([[2., 4.], [-2., -4.]]),
                     np.array([1., -2.]), np.array([3., -4.]))
        inputs = (weights, vbias, hbias, prev_gw, prev_gv, prev_gh)
        before = [a.copy() for a in (*inputs, *gradients)]
        expected = ([[1.29, -1.78], [3.07, -4.76]],
                    [.9, -1.], [1.6, -1.8],
                    [[.29, .22], [.07, -.76]], [.4, -.5], [.6, -.8])

        result = reference.parameter_update(
            *inputs, gradients, w_lr=.1, v_lr=.2, h_lr=.3,
            decay=.1, momentum=.5)
        repeated = reference.parameter_update(
            *inputs, gradients, w_lr=.1, v_lr=.2, h_lr=.3,
            decay=.1, momentum=.5)
        for actual, again, target in zip(result, repeated, expected):
            assert_allclose(actual, target, rtol=1e-6)
            assert_array_equal(actual, again)
            self.assertEqual(actual.dtype, np.dtype('float32'))
        for original, snapshot in zip((*inputs, *gradients), before):
            assert_array_equal(original, snapshot)

    def test_update_defaults_without_decay_or_momentum(self):
        inputs = (np.ones((5, 2)), np.ones(5), np.ones(2),
                  np.full((5, 2), 100.), np.full(5, 100.), np.full(2, 100.))
        gradients = (np.full((5, 2), 2.), np.full(5, 3.), np.full(2, 4.))
        result = reference.parameter_update(*inputs, gradients)
        for actual, target in zip(result[:3], [1.000042, 1.000075, 1.0001]):
            assert_allclose(actual, target, rtol=1e-7)
        for actual, target in zip(result[3:], [0.000042, 0.000075, 0.0001]):
            assert_allclose(actual, target, rtol=1e-7)

    def test_velocity_accumulates_steps_and_decays_with_zero_learning_rate(self):
        state = tuple(np.zeros(shape) for shape in [(1, 1), (1,), (1,)] * 2)
        gradients = tuple(np.full(shape, 2.) for shape in [(1, 1), (1,), (1,)])
        for rate, parameter, velocity in [(.1, .2, .2), (.1, .5, .3), (0., .65, .15)]:
            state = reference.parameter_update(
                *state, gradients, w_lr=rate, v_lr=rate, h_lr=rate, momentum=.5)
            for actual in state[:3]:
                assert_allclose(actual, parameter, rtol=1e-6)
            for actual in state[3:]:
                assert_allclose(actual, velocity, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
