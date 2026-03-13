import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model

logger = logging.getLogger(__name__)


class EmbeddingQualityExtractor(nn.Module):
    """Frozen wav2vec2 extractor for computing embedding quality scores.

    Teacher target for the embedding MetricGAN critic.
    All computation runs under no_grad — no gradient flows through this module.

    Supports multi-layer extraction: computes per-layer cosine similarity
    and averages across layers for the final quality score.
    """

    def __init__(self, model_name: str, layers: list[int]):
        super().__init__()
        self.layers = sorted(layers)
        max_layer = max(self.layers)

        model = Wav2Vec2Model.from_pretrained(
            model_name, output_hidden_states=True
        )
        model.encoder.layers = model.encoder.layers[:max_layer]
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.model.eval()
        logger.info(f"EmbeddingQualityExtractor: loaded wav2vec2 (layers={self.layers})")

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _normalize(self, waveform):
        """Zero-mean unit-variance normalization."""
        mean = waveform.mean(dim=-1, keepdim=True)
        std = waveform.std(dim=-1, keepdim=True).clamp(min=1e-7)
        return (waveform - mean) / std

    def _extract(self, waveform):
        """Extract frame embeddings at the target layers.

        Returns:
            list of [B, T_frames, D] tensors, one per layer.
        """
        waveform = self._normalize(waveform)
        out = self.model(waveform, output_hidden_states=True)
        return [out.hidden_states[l] for l in self.layers]

    @torch.no_grad()
    def extract_embeddings(self, waveform):
        """Extract frame embeddings (no_grad). Use to cache acoustic embeddings."""
        return self._extract(waveform)

    @torch.no_grad()
    def compute_frame_quality(self, acoustic, pair, z_ac=None):
        """Compute frame-level embedding quality scores q_t in [0, 1].

        Computes cosine similarity at each layer and averages across layers.

        Args:
            acoustic: [B, T_samples] reference waveform
            pair: [B, T_samples] comparison waveform (throat or enhanced)
            z_ac: list of [B, T_frames, D] pre-extracted acoustic embeddings
                  (optional, avoids redundant forward pass)

        Returns:
            q_t: [B, T_w2v] per-frame quality scores.
                 1.0 = identical to acoustic, 0.0 = max divergence.
        """
        # Align waveform lengths
        min_len = min(acoustic.shape[-1], pair.shape[-1])
        pair = pair[..., :min_len]

        if z_ac is None:
            acoustic = acoustic[..., :min_len]
            z_ac = self._extract(acoustic)

        z_pair = self._extract(pair)

        # All layers share the same T from one forward pass
        min_frames = min(z_ac[0].shape[1], z_pair[0].shape[1])

        # Average cosine similarity across layers
        q_sum = torch.zeros(z_ac[0].shape[0], min_frames, device=z_ac[0].device)
        for z_a, z_p in zip(z_ac, z_pair):
            q_sum += F.cosine_similarity(
                z_a[:, :min_frames, :], z_p[:, :min_frames, :], dim=-1)

        q_t = (q_sum / len(self.layers)).clamp(min=0.0, max=1.0)
        return q_t
