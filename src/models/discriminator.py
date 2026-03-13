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
        1. 2D encoder: stride=(2,1)×4 — halves freq 4 times, preserves time
        2. Freq collapse: AdaptiveMaxPool2d((1, None))
        3. Dilated 1D head: dilation 1,2,4,8 (RF ≈ 244ms)
        4. Time align: AdaptiveAvgPool1d(n_frames)
        5. Output: LearnableSigmoid(1)
    """

    def __init__(self, ndf=16, in_channel=2):
        super().__init__()

        # 2D encoder: stride=(2,1) halves freq, keeps time
        self.encoder = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, ndf, (4, 3), (2, 1), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 3), (2, 1), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(ndf * 2),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (4, 3), (2, 1), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(ndf * 4),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (4, 3), (2, 1), (1, 1), bias=False)),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(ndf * 8),
        )

        # Collapse frequency axis, keep time
        self.freq_collapse = nn.AdaptiveMaxPool2d((1, None))

        # Dilated 1D temporal head (no normalization — preserves temporal patterns)
        self.temporal = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv1d(ndf * 8, ndf * 4, 3, padding=1, dilation=1)),
            nn.PReLU(ndf * 4),
            nn.utils.spectral_norm(
                nn.Conv1d(ndf * 4, ndf * 4, 3, padding=2, dilation=2)),
            nn.PReLU(ndf * 4),
            nn.utils.spectral_norm(
                nn.Conv1d(ndf * 4, ndf * 2, 3, padding=4, dilation=4)),
            nn.PReLU(ndf * 2),
            nn.utils.spectral_norm(
                nn.Conv1d(ndf * 2, ndf * 2, 3, padding=8, dilation=8)),
            nn.PReLU(ndf * 2),
            nn.utils.spectral_norm(nn.Conv1d(ndf * 2, 1, 1)),
        )
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
        h = self.encoder(xy)                        # [B, 8*ndf, 12, T_disc]
        h = self.freq_collapse(h).squeeze(2)        # [B, 8*ndf, T_disc]
        h = self.temporal(h).squeeze(1)             # [B, T_disc]
        if n_frames is not None:
            h = nn.functional.adaptive_avg_pool1d(
                h.unsqueeze(1), n_frames).squeeze(1)  # [B, T_w2v]
        return self.output_act(h)
