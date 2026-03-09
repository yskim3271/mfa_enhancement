import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class EmbeddingLoss(nn.Module):
    """Cosine distance loss between enhanced and acoustic wav2vec2 embeddings.

    Frozen wav2vec2 extracts frame-level embeddings from both signals.
    The loss encourages enhanced audio embeddings to be close to acoustic ones.
    Gradients flow back through wav2vec2 to the enhancement model via enhanced input.
    """

    # wav2vec2 CNN feature extractor total stride: 5*2*2*2*2*2*2 = 320 samples
    WAV2VEC2_STRIDE = 320

    def __init__(self, model_name: str, layer: int, squared: bool = False,
                 silence_db: float = None, energy_floor_db: float = None):
        super().__init__()
        self.layer = layer
        self.squared = squared
        self.silence_db = silence_db
        self.energy_floor_db = energy_floor_db

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

    def _silence_mask(self, acoustic, n_frames):
        """Create mask to exclude silence frames based on acoustic energy.

        Args:
            acoustic: [B, T] reference waveform (detached)
            n_frames: number of wav2vec2 frames

        Returns:
            [B, n_frames] bool tensor, True = non-silence (keep)
        """
        stride = self.WAV2VEC2_STRIDE
        # Frame-level energy: unfold into [B, n_local_frames, stride]
        frames = acoustic.unfold(-1, stride, stride)
        energy = (frames ** 2).mean(dim=-1)  # [B, n_local_frames]
        energy = energy[:, :n_frames]

        # Pad if wav2vec2 produces more frames than simple striding
        if energy.shape[-1] < n_frames:
            pad = torch.zeros(
                energy.shape[0], n_frames - energy.shape[-1],
                device=energy.device,
            )
            energy = torch.cat([energy, pad], dim=-1)

        # Relative dB threshold: frames below (max_energy + silence_db) are silence
        max_energy = energy.max(dim=-1, keepdim=True).values.clamp(min=1e-10)
        energy_db = 10 * torch.log10(energy.clamp(min=1e-10) / max_energy)
        return energy_db > self.silence_db

    def _energy_weight(self, acoustic, n_frames):
        """Compute continuous energy-based weight (normalized dB).

        Maps frame energy to [0, 1] linearly in dB domain.
        Frames at 0 dB (max energy) get weight 1.0,
        frames at energy_floor_db get weight 0.0.

        Args:
            acoustic: [B, T] reference waveform (detached)
            n_frames: number of wav2vec2 frames

        Returns:
            [B, n_frames] float tensor in [0, 1]
        """
        stride = self.WAV2VEC2_STRIDE
        frames = acoustic.unfold(-1, stride, stride)
        energy = (frames ** 2).mean(dim=-1)  # [B, n_local_frames]
        energy = energy[:, :n_frames]

        if energy.shape[-1] < n_frames:
            pad = torch.zeros(
                energy.shape[0], n_frames - energy.shape[-1],
                device=energy.device,
            )
            energy = torch.cat([energy, pad], dim=-1)

        max_energy = energy.max(dim=-1, keepdim=True).values.clamp(min=1e-10)
        energy_db = 10 * torch.log10(energy.clamp(min=1e-10) / max_energy)
        # Linear in dB: floor_db → 0.0, 0 dB → 1.0
        weight = (energy_db - self.energy_floor_db) / (-self.energy_floor_db)
        return weight.clamp(min=0.0, max=1.0)

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

        # Apply energy weighting or silence masking
        if self.energy_floor_db is not None:
            with torch.no_grad():
                weight = self._energy_weight(acoustic.detach(), dist.shape[-1])
            return (dist * weight).sum() / weight.sum()
        elif self.silence_db is not None:
            with torch.no_grad():
                mask = self._silence_mask(acoustic.detach(), dist.shape[-1])
            if mask.any():
                return (dist * mask).sum() / mask.sum()

        return dist.mean()
