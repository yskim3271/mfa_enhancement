import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

logger = logging.getLogger(__name__)


class EmbeddingLoss(nn.Module):
    """Cosine distance loss between enhanced and acoustic wav2vec2 embeddings.

    Frozen wav2vec2 extracts frame-level embeddings from both signals.
    The loss encourages enhanced audio embeddings to be close to acoustic ones.
    Gradients flow back through wav2vec2 to the enhancement model via enhanced input.
    """

    def __init__(self, model_name: str, layer: int,
                 risk_calibrated: bool = False, risk_beta: tuple = (-4.85, 6.72),
                 local_window: int = 0):
        super().__init__()
        self.layer = layer
        self.risk_calibrated = risk_calibrated
        self.local_window = local_window

        # Store risk calibration coefficients as buffer (saved in checkpoint)
        self.register_buffer("risk_beta0", torch.tensor(float(risk_beta[0])))
        self.register_buffer("risk_beta1", torch.tensor(float(risk_beta[1])))

        model = Wav2Vec2Model.from_pretrained(
            model_name, output_hidden_states=True
        )
        # Truncate encoder to only compute up to the target layer
        model.encoder.layers = model.encoder.layers[:layer]

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.model.eval()

    def train(self, mode=True):
        # Keep wav2vec2 always in eval mode
        super().train(mode)
        self.model.eval()
        return self

    def _normalize(self, waveform):
        """Zero-mean unit-variance normalization (differentiable)."""
        mean = waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True).clamp(min=1e-7)
        return (waveform - mean) / std

    def _extract(self, waveform):
        """Extract frame embeddings at the target layer."""
        waveform = self._normalize(waveform)
        out = self.model(waveform, output_hidden_states=True)
        # hidden_states[0] = CNN output, [1..N] = transformer layers
        return out.hidden_states[self.layer]

    def _compute_local_window_dist(self, emb_enhanced, emb_acoustic):
        """Compute cosine distance with local-window alignment (hard min).

        For each enhanced frame t, find the minimum cosine distance among
        acoustic frames in [t-K, t+K], tolerating minor boundary jitter.

        Args:
            emb_enhanced: [B, T, D] enhanced embeddings
            emb_acoustic: [B, T, D] acoustic embeddings

        Returns:
            [B, T] minimum cosine distance per frame
        """
        K = self.local_window
        T = emb_enhanced.shape[1]
        dists = []
        for delta in range(-K, K + 1):
            if delta == 0:
                ac_shifted = emb_acoustic
            elif delta > 0:
                ac_shifted = F.pad(emb_acoustic[:, delta:, :], (0, 0, 0, delta))
            else:
                ac_shifted = F.pad(emb_acoustic[:, :delta, :], (0, 0, -delta, 0))
            d = 1 - F.cosine_similarity(emb_enhanced, ac_shifted[:, :T, :], dim=-1)
            dists.append(d)
        return torch.stack(dists, dim=-1).min(dim=-1).values

    def forward(self, enhanced, acoustic):
        """Compute mean cosine distance between enhanced and acoustic embeddings.

        Args:
            enhanced: [B, T] waveform from enhancement model (gradient flows)
            acoustic: [B, T] reference waveform (detached, no gradient)
        """
        # Align lengths
        min_len = min(enhanced.shape[-1], acoustic.shape[-1])
        enhanced = enhanced[..., :min_len]
        acoustic = acoustic[..., :min_len]

        # Enhanced: gradient preserved for backprop to enhancement model
        emb_enhanced = self._extract(enhanced)

        # Acoustic: no gradient needed
        with torch.no_grad():
            emb_acoustic = self._extract(acoustic.detach())

        # Cosine distance
        if self.local_window > 0:
            dist = self._compute_local_window_dist(emb_enhanced, emb_acoustic)
        else:
            cos_sim = F.cosine_similarity(emb_enhanced, emb_acoustic, dim=-1)
            dist = 1 - cos_sim

        # Transform: risk calibration
        if self.risk_calibrated:
            dist = torch.sigmoid(self.risk_beta0 + self.risk_beta1 * dist)

        return dist.mean()
