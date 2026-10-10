"""Deterministic NumPy reference for cfrbm/rbm.py math.

This includes velocity momentum (F10), observed-only phase masks (F11), and
softmax visible probabilities (F12). Supply binary hidden and one-hot visible
samples explicitly; no random generator or tensor backend is needed. Arrays use
the same batch/visible/hidden axes as CFRBM. The update returns new parameters
and velocity memory without mutating inputs, matching simultaneous shared updates.
"""

import numpy as np


def outer(x, y):
    return x[:, :, np.newaxis] * y[:, np.newaxis, :]


def prop_up(vis, weights, hbias):
    return 1.0 / (1.0 + np.exp(-(np.dot(vis, weights) + hbias)))


def prop_down(h, weights, vbias, k=5):
    logits = np.dot(h, weights.T) + vbias
    groups = logits.reshape(-1, k)
    exponentials = np.exp(groups - groups.max(axis=1, keepdims=True))
    return (exponentials / exponentials.sum(axis=1, keepdims=True)).reshape(logits.shape)


def sample_visible(h, weights, vbias, visible_sample, k=5):
    """Return a supplied categorical one-hot sample and softmax probabilities."""
    activations = prop_down(h, weights, vbias, k=k)
    return np.array(visible_sample, copy=True), activations


def gradient(v1, h1, v2, h2p, masks):
    """Return (weight, visible-bias, hidden-bias) batch-mean gradients."""
    observed = masks[:, :, np.newaxis]
    gw = ((outer(v1, h1) - outer(v2, h2p)) * observed).mean(axis=0)
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
