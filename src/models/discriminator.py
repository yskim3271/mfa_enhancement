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


class DilatedResBlock1d(nn.Module):
    """Two dilated Conv1d layers with a residual skip (1x1 proj if channels differ)."""

    def __init__(self, in_ch, out_ch, dilations):
        super().__init__()
        layers = []
        ch = in_ch
        for d in dilations:
            layers.append(nn.utils.spectral_norm(
                nn.Conv1d(ch, out_ch, 3, padding=d, dilation=d)))
            layers.append(nn.PReLU(out_ch))
            ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.skip = (nn.utils.spectral_norm(nn.Conv1d(in_ch, out_ch, 1))
                     if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        return self.layers(x) + self.skip(x)


class FrameLevelEmbeddingCritic(nn.Module):
    """Frame-level embedding critic for MetricGAN.

    Outputs per-frame quality predictions [B, n_frames] instead of
    a single scalar, enabling temporally differentiated gradients.

    Architecture:
        1. 2D encoder: stride=(2,1)×4 — halves freq 4 times, preserves time
        2. Freq collapse: (3,1) conv ×3 — 12→6→3→1, absorbed in encoder
        3. Dilated 1D head: residual blocks (d=1,2) + (d=4,8) with 1x1 skip
        4. Time align: AdaptiveAvgPool1d(n_frames)
        5. Output: LearnableSigmoid(1)
    """

    def __init__(self, ndf=16, in_channel=2):
        super().__init__()

        # 2D encoder: stride=(2,1) halves freq, keeps time
        # then (3,1) conv collapses remaining freq: 12→6→3→1
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
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 8, ndf * 8, (3, 1), (2, 1), (1, 0))),
            nn.PReLU(ndf * 8),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 8, ndf * 8, (3, 1), (2, 1), (1, 0))),
            nn.PReLU(ndf * 8),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 8, ndf * 8, (3, 1), (3, 1), (0, 0))),
            nn.PReLU(ndf * 8),
        )

        # Simplified Channel Attention (SCA): recalibrate encoder channels
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(ndf * 8, ndf * 8, kernel_size=1),
        )

        # Dilated 1D temporal head with residual connections
        self.temporal = nn.Sequential(
            DilatedResBlock1d(ndf * 8, ndf * 4, [1, 2]),
            DilatedResBlock1d(ndf * 4, ndf * 2, [4, 8]),
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
        h = self.encoder(xy)                        # [B, 8*ndf, 1, T_disc]
        assert h.shape[2] == 1, f"Expected freq=1 after encoder, got {h.shape[2]}"
        h = h.squeeze(2)                            # [B, 8*ndf, T_disc]
        h = h * self.sca(h)                         # SCA: channel attention
        h = self.temporal(h).squeeze(1)             # [B, T_disc]
        if n_frames is not None:
            h = nn.functional.adaptive_avg_pool1d(
                h.unsqueeze(1), n_frames).squeeze(1)  # [B, T_w2v]
        return self.output_act(h)
