"""PyTorch RBM with contrastive divergence and velocity momentum."""

import numpy as np
import torch

from .device import select_device


class CFRBM:
    def __init__(self, num_visible, num_hidden, initial_v=None,
                 initial_weigths=None, debug=False, device=None):
        """Create float32 state on the selected device (CUDA, MPS, then CPU).

        NumPy's global RNG still initializes weights from normal(0, 0.1).
        Sampling has its own torch generator seeded with 17; its sequence is
        not intended to match the original backend's RNG. ``debug`` remains
        accepted for compatibility, but no symbolic compiler settings are needed.
        """
        self.device = select_device(device)
        self.dim = (num_visible, num_hidden)
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        if initial_weigths:
            initial_weights = np.load('{}.W.npy'.format(initial_weigths))
            initial_hbias = np.load('{}.h.npy'.format(initial_weigths))
            initial_vbias = np.load('{}.b.npy'.format(initial_weigths))
        else:
            initial_weights = np.array(np.random.normal(0, 0.1, size=self.dim),
                                       dtype=np.float32)
            initial_hbias = np.zeros(num_hidden, dtype=np.float32)
            initial_vbias = (initial_v if initial_v is not None
                             else np.zeros(num_visible, dtype=np.float32))

        self.weights = torch.tensor(initial_weights, dtype=torch.float32, device=self.device)
        self.vbias = torch.tensor(initial_vbias, dtype=torch.float32, device=self.device)
        self.hbias = torch.tensor(initial_hbias, dtype=torch.float32, device=self.device)
        # Historical prev_g* names now hold parameter velocities, not gradients.
        self.prev_gw = torch.zeros_like(self.weights)
        self.prev_gh = torch.zeros_like(self.hbias)
        self.prev_gv = torch.zeros_like(self.vbias)
        self.generator = torch.Generator(device=self.device).manual_seed(17)

    def prop_up(self, vis):
        return torch.sigmoid(vis @ self.weights + self.hbias)

    def sample_hidden(self, vis):
        activations = self.prop_up(vis)
        return torch.bernoulli(activations, generator=self.generator), activations

    def prop_down(self, h):
        return torch.sigmoid(h @ self.weights.T + self.vbias)

    def sample_visible(self, h, k=5):
        activations = self.prop_down(h)
        groups = activations.reshape(-1, k)
        activations = (groups / groups.sum(dim=1, keepdim=True)).reshape_as(activations)
        return torch.bernoulli(activations, generator=self.generator), activations

    def contrastive_divergence_1(self, v1):
        h1, _ = self.sample_hidden(v1)
        v2, v2a = self.sample_visible(h1)
        h2, h2a = self.sample_hidden(v2)
        return v1, h1, v2, v2a, h2, h2a

    def gradient(self, v1, h1, v2, h2p, masks):
        # h1 is binary, so h1 squared in the legacy positive phase is h1.
        # Retain the extra negative-phase h1 factor until F11.
        positive, negative = v1 * masks, v2 * masks
        gw = (positive.T @ h1 - negative.T @ (h2p * h1)) / v1.shape[0]
        gv = (positive - negative).mean(dim=0)
        gh = (h1 - h2p).mean(dim=0)
        return gw, gv, gh

    def cdk_fun(self, vis, masks, k=1, w_lr=0.000021, v_lr=0.000025,
                h_lr=0.000025, decay=0.0000, momentum=0.0):
        """Return train(batch, masks); factory inputs are compatibility placeholders.

        As in the legacy implementation, CD-k computes its gradient from the
        final transition. Momentum retains the learning-rate-scaled parameter
        step across calls, including calls from a new epoch factory.
        """
        @torch.no_grad()
        def train(batch, batch_masks):
            v1, h1, v2, _, _, h2a = self.contrastive_divergence_1(batch)
            for _ in range(k - 1):
                v1, h1, v2, _, _, h2a = self.contrastive_divergence_1(v2)
            W, V, H = self.gradient(v1, h1, v2, h2a, batch_masks)
            if decay:
                W = W - decay * self.weights

            # Weight decay is part of the gradient before learning-rate scaling.
            # Preserve state identities and retain the actual step for momentum.
            self.prev_gw.copy_(momentum * self.prev_gw + w_lr * W)
            self.prev_gv.copy_(momentum * self.prev_gv + v_lr * V)
            self.prev_gh.copy_(momentum * self.prev_gh + h_lr * H)
            self.weights.add_(self.prev_gw)
            self.vbias.add_(self.prev_gv)
            self.hbias.add_(self.prev_gh)

        return train

    def predict(self, v1=None):
        """Return a callable producing normalized visible probabilities.

        The optional factory input preserves predict(vis) calls. Supply the
        actual tensor batch to the returned callable; prediction is stochastic.
        """
        @torch.no_grad()
        def predict_batch(batch):
            h1, _ = self.sample_hidden(batch)
            _, probabilities = self.sample_visible(h1)
            return probabilities

        return predict_batch
