import numpy as np
import torch
import torch.nn.functional as F


def _smooth_signal(x, kernel_size):
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smoothed = np.empty_like(x, dtype=np.float32)
    for ch in range(x.shape[0]):
        padded = np.pad(x[ch], (pad, pad), mode="reflect")
        smoothed[ch] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _normalize_segment(segment):
    segment = segment.astype(np.float32, copy=False)
    segment = segment - np.mean(segment, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(segment ** 2) + 1e-8)
    return segment / max(rms, 1e-4)


def _select_window_start(signal, segment_length, rng):
    total_length = signal.shape[-1]
    if total_length <= segment_length:
        return 0

    energy = np.sum(signal ** 2, axis=0)
    window_energy = np.convolve(energy, np.ones(segment_length, dtype=np.float32), mode="valid")
    topk = max(1, min(window_energy.shape[0], window_energy.shape[0] // 5))
    candidate_idx = np.argpartition(window_energy, -topk)[-topk:]
    return int(candidate_idx[rng.integers(0, len(candidate_idx))])


def build_device_residual_bank(
    x,
    y,
    segment_length,
    max_templates=256,
    per_device=32,
    smooth_kernel=33,
    seed=2023,
):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    segment_length = int(segment_length)
    per_device = max(1, int(per_device))
    max_templates = max(1, int(max_templates))

    templates = []
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        if indices.size == 0:
            continue
        rng.shuffle(indices)
        take = min(indices.size, per_device)
        for idx in indices[:take]:
            sample = np.asarray(x[idx], dtype=np.float32)
            residual = sample - _smooth_signal(sample, smooth_kernel)
            start = _select_window_start(residual, segment_length, rng)
            end = min(start + segment_length, residual.shape[-1])
            segment = residual[:, start:end]
            if segment.shape[-1] < segment_length:
                pad_width = segment_length - segment.shape[-1]
                segment = np.pad(segment, ((0, 0), (0, pad_width)), mode="constant")
            templates.append(_normalize_segment(segment))

    if not templates:
        raise ValueError("Failed to build device residual bank: no templates extracted.")

    bank = np.stack(templates, axis=0)
    if bank.shape[0] > max_templates:
        selected = rng.choice(bank.shape[0], size=max_templates, replace=False)
        bank = bank[selected]
    return torch.tensor(bank, dtype=torch.float32)


def log_psd(batch, eps=1e-6):
    spectrum = torch.fft.rfft(batch, dim=-1)
    power = spectrum.real.pow(2) + spectrum.imag.pow(2)
    power = power / max(batch.shape[-1], 1)
    return torch.log(power + eps)


def log_spectrogram(batch, n_fft=32, hop_length=8, win_length=32, eps=1e-6):
    batch = batch.reshape(batch.shape[0] * batch.shape[1], batch.shape[2])
    window = torch.hann_window(win_length, device=batch.device, dtype=batch.dtype)
    spec = torch.stft(
        batch,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(spec)
    freq_bins = mag.shape[-2]
    time_bins = mag.shape[-1]
    return torch.log(mag + eps).reshape(-1, 2, freq_bins, time_bins)


def device_residual_matching_loss(
    trigger_pattern,
    residual_bank,
    batch_size=32,
    match_mode="spectrogram",
    n_fft=32,
    hop_length=8,
    win_length=32,
):
    if residual_bank is None or residual_bank.numel() == 0:
        return trigger_pattern.new_tensor(0.0)

    bank = residual_bank.to(device=trigger_pattern.device, dtype=trigger_pattern.dtype)
    sample_count = min(int(batch_size), bank.shape[0])
    if sample_count <= 0:
        return trigger_pattern.new_tensor(0.0)

    indices = torch.randperm(bank.shape[0], device=trigger_pattern.device)[:sample_count]
    ref_batch = bank.index_select(0, indices)
    if match_mode == "spectrogram":
        trigger_feat = log_spectrogram(
            trigger_pattern.unsqueeze(0),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        ).expand(sample_count, -1, -1, -1)
        ref_feat = log_spectrogram(
            ref_batch,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
    else:
        trigger_feat = log_psd(trigger_pattern.unsqueeze(0)).expand(sample_count, -1, -1)
        ref_feat = log_psd(ref_batch)
    return F.mse_loss(trigger_feat, ref_feat)
