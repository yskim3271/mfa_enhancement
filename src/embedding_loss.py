import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class EmbeddingLoss(nn.Module):
    """Cosine distance loss between enhanced and acoustic wav2vec2 embeddings.

    Frozen wav2vec2 extracts frame-level embeddings from both signals.
    The loss encourages enhanced audio embeddings to be close to acoustic ones.
    Gradients flow back through wav2vec2 to the enhancement model via enhanced input.
    """

    def __init__(self, model_name: str, layer: int, squared: bool = False):
        super().__init__()
        self.layer = layer
        self.squared = squared

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

        # Cosine distance: 1 - cosine_similarity
        cos_sim = nn.functional.cosine_similarity(
            emb_enhanced, emb_acoustic, dim=-1
        )
        dist = 1 - cos_sim
        if self.squared:
            dist = dist ** 2
        return dist.mean()
