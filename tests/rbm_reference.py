"""Deterministic NumPy reference for cfrbm/rbm.py math.

This includes velocity momentum (F10), retaining the F11–F12 quirks. Supply binary
samples explicitly; no random generator or tensor backend is needed. Arrays use
the same batch/visible/hidden axes as CFRBM. The update returns new parameters
and velocity memory without mutating inputs, matching simultaneous shared updates.
"""

import numpy as np


def outer(x, y):
    return x[:, :, np.newaxis] * y[:, np.newaxis, :]


def prop_up(vis, weights, hbias):
    return 1.0 / (1.0 + np.exp(-(np.dot(vis, weights) + hbias)))


def prop_down(h, weights, vbias):
    return 1.0 / (1.0 + np.exp(-(np.dot(h, weights.T) + vbias)))


def sample_visible(h, weights, vbias, visible_sample, k=5):
    """Return a supplied Bernoulli sample and the normalized sigmoid values.

    Each consecutive group of k activations sums to one, but its independent
    binary samples need not be one-hot. This is deliberately not a softmax.
    """
    activations = prop_down(h, weights, vbias)
    k_ones = np.ones(k)
    partitions = (
        activations.reshape((-1, k)).sum(axis=1).reshape((-1, 1)) * k_ones
    )
    activations = activations / partitions.reshape(activations.shape)
    return np.array(visible_sample, copy=True), activations


def gradient(v1, h1, v2, h2p, masks):
    """Return (weight, visible-bias, hidden-bias) batch-mean gradients."""
    v1h1_mask = outer(masks, h1)
    # Preserve both h1 factors in the positive phase and h1 in the negative
    # phase. Binary h1 makes its positive-phase square equal to itself.
    gw = ((outer(v1, h1) * v1h1_mask)
          - (outer(v2, h2p) * v1h1_mask)).mean(axis=0)
    gv = ((v1 * masks) - (v2 * masks)).mean(axis=0)
    gh = (h1 - h2p).mean(axis=0)
    return gw, gv, gh


def parameter_update(weights, vbias, hbias, prev_gw, prev_gv, prev_gh,
                     gradients, w_lr=0.000021, v_lr=0.000025,
                     h_lr=0.000025, decay=0.0, momentum=0.0):
    """Return parameters and velocities (historical prev_g* state names).

    Decay applies only to weights, before learning-rate scaling. Return new
    float32 arrays without mutating the supplied parameters or velocities.
    """
    W, V, H = gradients
    if decay:
        W = W - decay * weights
    velocities = tuple(np.asarray(momentum * previous + rate * gradient,
                                  dtype=np.float32)
                       for previous, rate, gradient in
                       zip((prev_gw, prev_gv, prev_gh), (w_lr, v_lr, h_lr),
                           (W, V, H)))
    parameters = tuple(np.asarray(parameter + velocity, dtype=np.float32)
                       for parameter, velocity in
                       zip((weights, vbias, hbias), velocities))
    return parameters + velocities
