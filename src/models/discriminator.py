import torch
import torch.nn as nn


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class MetricGAN_Discriminator(nn.Module):
    def __init__(self, ndf=16, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)


class FrameLevelEmbeddingCritic(nn.Module):
    """Frame-level embedding critic for MetricGAN.

    Outputs per-frame quality predictions [B, n_frames] instead of
    a single scalar, enabling temporally differentiated gradients.

    Architecture:
        1. 2D encoder: 5 blocks — k(4,3)/s(2,1)×2 then k(8,3)/s(4,1)×3
        2. Channel collapse: Conv1d(4*ndf, 1, 1)
        3. Time align: AdaptiveAvgPool1d(n_frames)
        4. Output: LearnableSigmoid(1)
    """

    def __init__(self, ndf=16, in_channel=2):
        super().__init__()

        # 2D encoder: halves freq each block, preserves time
        self.encoder = nn.Sequential(
            # Block 1: F:257→128, C:in→ndf  (k=4,s=2)
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 3), (2, 1), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            # Block 2: F:128→64, C:ndf→2ndf  (k=4,s=2)
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 3), (2, 1), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(ndf * 2),
            # Block 3: F:64→16, C:2ndf→4ndf  (k=8,s=4)
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (8, 3), (4, 1), (2, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(ndf * 4),
            # Block 4: F:16→4, C:4ndf→4ndf  (k=8,s=4)
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 4, (8, 3), (4, 1), (2, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(ndf * 4),
            # Block 5: F:4→1, C:4ndf→4ndf  (k=8,s=4)
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 4, (8, 3), (4, 1), (2, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(ndf * 4),
        )

        # Channel collapse: 4*ndf → 1
        self.proj = nn.utils.spectral_norm(nn.Conv1d(ndf * 4, 1, 1))
        self.output_act = LearnableSigmoid(1)

    def forward(self, x, y, n_frames=None):
        """
        Args:
            x: [B, 1, F, T_disc] reference magnitude spectrogram
            y: [B, 1, F, T_disc] comparison magnitude spectrogram
            n_frames: target output frames (T_w2v). None → T_disc resolution.

        Returns:
            [B, n_frames] quality scores in [0, 1]
        """
        xy = torch.cat([x, y], dim=1)              # [B, 2, F, T_disc]
        h = self.encoder(xy)                        # [B, 4*ndf, 1, T_disc]
        h = h.squeeze(2)                            # [B, 4*ndf, T_disc]
        h = self.proj(h)                            # [B, 1, T_disc]
        if n_frames is not None:
            h = nn.functional.adaptive_avg_pool1d(
                h, n_frames)                        # [B, 1, T_w2v]
        h = h.squeeze(1)                            # [B, T]
        return self.output_act(h)
