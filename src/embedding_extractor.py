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
    """

    def __init__(self, model_name: str, layer: int, shared_model=None):
        super().__init__()
        self.layer = layer

        if shared_model is not None:
            self.model = shared_model
            logger.info("EmbeddingQualityExtractor: reusing shared wav2vec2 model")
        else:
            model = Wav2Vec2Model.from_pretrained(
                model_name, output_hidden_states=True
            )
            model.encoder.layers = model.encoder.layers[:layer]
            for param in model.parameters():
                param.requires_grad = False
            self.model = model
            self.model.eval()
            logger.info(f"EmbeddingQualityExtractor: loaded own wav2vec2 (layer={layer})")

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
        """Extract frame embeddings at the target layer."""
        waveform = self._normalize(waveform)
        out = self.model(waveform, output_hidden_states=True)
        return out.hidden_states[self.layer]

    @torch.no_grad()
    def extract_embeddings(self, waveform):
        """Extract frame embeddings (no_grad). Use to cache acoustic embeddings."""
        return self._extract(waveform)

    @torch.no_grad()
    def compute_frame_quality(self, acoustic, pair, z_ac=None):
        """Compute frame-level embedding quality scores q_t in [0, 1].

        Args:
            acoustic: [B, T_samples] reference waveform
            pair: [B, T_samples] comparison waveform (throat or enhanced)
            z_ac: [B, T_frames, D] pre-extracted acoustic embeddings (optional,
                  avoids redundant forward pass when calling multiple times
                  with the same acoustic reference)

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

        # Align frame lengths (may differ by 1 due to conv stride)
        min_frames = min(z_ac.shape[1], z_pair.shape[1])
        z_ac = z_ac[:, :min_frames, :]
        z_pair = z_pair[:, :min_frames, :]

        # Frame-level quality: cosine similarity clamped to [0, 1]
        q_t = F.cosine_similarity(z_ac, z_pair, dim=-1).clamp(min=0.0, max=1.0)  # [B, T_w2v]

        return q_t
