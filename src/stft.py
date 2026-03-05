import torch

_hann_cache = {}

def _get_hann_window(win_size, device):
    key = (win_size, device)
    if key not in _hann_cache:
        _hann_cache[key] = torch.hann_window(win_size, device=device)
    return _hann_cache[key]

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True, stack_dim=-1):

    hann_window = _get_hann_window(win_size, y.device)
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1)+(1e-9))
    # Add epsilon to prevent gradient explosion in atan2 backward
    # When magnitude is very small (near silence), atan2 gradient can explode
    pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-8, stft_spec[:, :, :, 0] + 1e-8)
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag*torch.cos(pha), mag*torch.sin(pha)), dim=stack_dim)

    return mag, pha, com

def mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Magnitude Decompression
    mag = torch.pow(mag, (1.0/compress_factor))
    com = torch.complex(mag*torch.cos(pha), mag*torch.sin(pha))
    hann_window = _get_hann_window(win_size, com.device)

    if center:
        wav = torch.istft(com, n_fft=n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=True)
    else:
        # torch.istft(center=False) fails COLA check with Hann window.
        # Use manual OLA reconstruction instead.
        frames = torch.fft.irfft(com.transpose(-2, -1), n=n_fft)  # [B, T, n_fft]
        frames = frames[..., :win_size]  # [B, T, win_size]
        frames = frames * hann_window
        window_sq = hann_window * hann_window

        B, T = frames.shape[:2]
        output_len = (T - 1) * hop_size + win_size

        buf = torch.zeros(B, output_len, device=com.device, dtype=frames.dtype)
        norm = torch.zeros(output_len, device=com.device, dtype=frames.dtype)

        for t in range(T):
            start = t * hop_size
            buf[:, start:start + win_size] += frames[:, t]
            norm[start:start + win_size] += window_sq

        safe_mask = norm > 1e-8
        buf[:, safe_mask] = buf[:, safe_mask] / norm[safe_mask]
        wav = buf

    return wav
