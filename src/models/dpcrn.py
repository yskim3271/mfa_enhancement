import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Module):
    """Conv2d with causal padding on time axis, custom padding on freq axis."""

    def __init__(self, in_ch, out_ch, kernel_size, stride, freq_pad):
        super().__init__()
        # freq_pad: (left, right), time: causal = (kernel_t - 1, 0)
        self.pad = (freq_pad[0], freq_pad[1], kernel_size[0] - 1, 0)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU(out_ch)

    def forward(self, x):
        # x: [B, C, T, F]
        x = F.pad(x, self.pad)
        return self.act(self.bn(self.conv(x)))


class InstantLayerNorm(nn.Module):
    """Per-frame LayerNorm over (channel, frequency).

    Equivalent to TF LayerNormalization(axis=[-1,-2]) on [B, T, F, C].
    Input: [B, C, T, F]. Normalizes over (C, F) per (B, T) position.
    """

    def __init__(self, channels, freq_dim):
        super().__init__()
        self.norm = nn.LayerNorm([channels, freq_dim])

    def forward(self, x):
        # [B, C, T, F] -> [B, T, C, F] -> norm -> [B, C, T, F]
        x = x.permute(0, 2, 1, 3)
        x = self.norm(x)
        x = x.permute(0, 2, 1, 3)
        return x


class DPRNNBlock(nn.Module):
    """Dual-Path RNN block: intra-chunk (freq) + inter-chunk (time)."""

    def __init__(self, channels=128, width=50, hidden=128):
        super().__init__()
        # Intra-chunk: bidirectional LSTM along frequency axis
        self.intra_rnn = nn.LSTM(
            channels, hidden // 2, batch_first=True, bidirectional=True,
        )
        self.intra_fc = nn.Linear(hidden, channels)
        self.intra_ln = InstantLayerNorm(channels, width)

        # Inter-chunk: unidirectional LSTM along time axis
        self.inter_rnn = nn.LSTM(
            channels, hidden, batch_first=True, bidirectional=False,
        )
        self.inter_fc = nn.Linear(hidden, channels)
        self.inter_ln = InstantLayerNorm(channels, width)

    def forward(self, x):
        # x: [B, C, T, F]
        B, C, T, F_ = x.shape

        # --- Intra-chunk (frequency direction) ---
        # [B, C, T, F] -> [B*T, F, C]
        intra_in = x.permute(0, 2, 3, 1).reshape(B * T, F_, C)
        intra_out, _ = self.intra_rnn(intra_in)
        intra_out = self.intra_fc(intra_out)
        intra_out = intra_out.reshape(B, T, F_, C).permute(0, 3, 1, 2)
        intra_out = self.intra_ln(intra_out) + x

        # --- Inter-chunk (time direction) ---
        # [B, C, T, F] -> [B*F, T, C]
        inter_in = intra_out.permute(0, 3, 2, 1).reshape(B * F_, T, C)
        inter_out, _ = self.inter_rnn(inter_in)
        inter_out = self.inter_fc(inter_out)
        inter_out = inter_out.reshape(B, F_, T, C).permute(0, 3, 2, 1)
        inter_out = self.inter_ln(inter_out) + intra_out

        return inter_out


class DPCRN(nn.Module):
    """Dual-Path Convolution Recurrent Network for speech enhancement.

    Encoder-decoder with dual-path RNN bottleneck and CRM output.
    Reference: Le et al., "DPCRN", Interspeech 2021.
    """

    def __init__(
        self,
        enc_channels=(32, 32, 32, 64, 128),
        num_dprnn=2,
        dprnn_hidden=128,
        n_freqs=201,
    ):
        super().__init__()
        self.enc_channels = enc_channels

        # Input normalization: instant LN over (channel=2, freq) per time frame
        self.input_norm = nn.LayerNorm([2, n_freqs])

        # --- Encoder (5 layers) ---
        # Specs: (in_ch, out_ch, kernel, stride, freq_pad=(left, right))
        enc_specs = [
            (2,              enc_channels[0], (2, 5), (1, 2), (0, 2)),
            (enc_channels[0], enc_channels[1], (2, 3), (1, 2), (0, 1)),
            (enc_channels[1], enc_channels[2], (2, 3), (1, 1), (1, 1)),
            (enc_channels[2], enc_channels[3], (2, 3), (1, 1), (1, 1)),
            (enc_channels[3], enc_channels[4], (2, 3), (1, 1), (1, 1)),
        ]
        self.encoder = nn.ModuleList([
            CausalConv2d(*spec) for spec in enc_specs
        ])

        # --- DPRNN bottleneck ---
        # F after encoder: 201 -> 100 -> 50
        self.dprnn = nn.ModuleList([
            DPRNNBlock(channels=enc_channels[-1], width=50, hidden=dprnn_hidden)
            for _ in range(num_dprnn)
        ])

        # --- Decoder (5 layers with skip connections) ---
        C = enc_channels
        # DeConv1-3: stride=(1,1), freq preserved
        # DeConv4: stride=(1,2), freq 50->100
        # DeConv5: stride=(1,2), freq 100->201
        # Time: kernel_t=2, stride_t=1, no padding -> crop 1 from end
        dec_specs = [
            # (in_ch_after_cat, out_ch, kernel, stride, freq_padding)
            (C[4] * 2, C[3], (2, 3), (1, 1), 1),
            (C[3] * 2, C[2], (2, 3), (1, 1), 1),
            (C[2] * 2, C[1], (2, 3), (1, 1), 1),
            (C[1] * 2, C[0], (2, 3), (1, 2), 1),
        ]
        self.decoder = nn.ModuleList()
        for in_ch, out_ch, ks, st, fp in dec_specs:
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, ks, stride=st,
                                   padding=(0, fp), output_padding=(0, st[1] - 1)),
                nn.BatchNorm2d(out_ch),
                nn.PReLU(out_ch),
            ))

        # DeConv5: upsample freq 100->201, no activation
        self.decoder_final = nn.ConvTranspose2d(
            C[0] * 2, 2, (2, 5), stride=(1, 2), padding=(0, 0),
        )

    def forward(self, noisy_com):
        """
        Args:
            noisy_com: [B, F, T, 2] compressed complex spectrogram
        Returns:
            est_mag: [B, F, T]
            est_pha: [B, F, T]
            est_com: [B, F, T, 2]
        """
        # [B, F, T, 2] -> [B, 2, T, F]
        x = noisy_com.permute(0, 3, 2, 1)

        # Input normalization: LN over (C=2, F) per time frame
        x = x.permute(0, 2, 1, 3)      # [B, T, 2, F]
        x = self.input_norm(x)
        x = x.permute(0, 2, 1, 3)      # [B, 2, T, F]

        # --- Encoder ---
        skips = []
        for enc_layer in self.encoder:
            x = enc_layer(x)
            skips.append(x)
        # x: [B, 128, T, 50]

        # --- DPRNN ---
        for block in self.dprnn:
            x = block(x)

        # --- Decoder ---
        for i, dec_layer in enumerate(self.decoder):
            skip = skips[len(skips) - 1 - i]
            x = torch.cat([x, skip], dim=1)
            x = dec_layer(x)
            x = x[:, :, :-1]  # crop extra time sample from kernel_t=2

        # DeConv5
        x = torch.cat([x, skips[0]], dim=1)
        x = self.decoder_final(x)
        x = x[:, :, :-1, :-2]  # crop time and freq

        # --- Mapping: decoder output is the estimated clean spectrogram ---
        est_r = x[:, 0]
        est_i = x[:, 1]

        # [B, T, F] -> [B, F, T]
        est_r = est_r.transpose(1, 2)
        est_i = est_i.transpose(1, 2)

        est_mag = torch.sqrt(est_r ** 2 + est_i ** 2 + 1e-8)
        est_pha = torch.atan2(est_i + 1e-8, est_r + 1e-8)
        est_com = torch.stack((est_r, est_i), dim=-1)

        return est_mag, est_pha, est_com
