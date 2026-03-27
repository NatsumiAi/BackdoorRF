import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F


def _normalize_template(template):
    template = template.astype(np.float32, copy=False)
    template = template - np.mean(template, axis=1, keepdims=True)
    rms = np.sqrt(np.mean(template ** 2) + 1e-8)
    return template / max(rms, 1e-4)


def _smooth_template(template, kernel_size):
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return template.astype(np.float32, copy=False)
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smoothed = np.empty_like(template, dtype=np.float32)
    for ch in range(template.shape[0]):
        padded = np.pad(template[ch], (pad, pad), mode="reflect")
        smoothed[ch] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _build_frequency_mask(length, low_ratio, high_ratio):
    freq_bins = length // 2 + 1
    mask = np.zeros(freq_bins, dtype=np.float32)
    low_idx = max(1, int(round(low_ratio * (freq_bins - 1))))
    high_idx = max(low_idx + 1, int(round(high_ratio * (freq_bins - 1))))
    high_idx = min(high_idx, freq_bins - 1)
    mask[low_idx:high_idx + 1] = 1.0
    if mask.sum() == 0:
        mask[1:min(freq_bins, 4)] = 1.0
    return mask


def _generate_band_limited_noise(length, rng, low_ratio, high_ratio):
    mask = _build_frequency_mask(length, low_ratio, high_ratio)
    channels = []
    for _ in range(2):
        noise = rng.standard_normal(length).astype(np.float32)
        spectrum = np.fft.rfft(noise)
        phases = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=spectrum.shape[0]))
        shaped = spectrum * mask * phases
        channel = np.fft.irfft(shaped, n=length).astype(np.float32)
        channels.append(channel)
    return np.stack(channels, axis=0)


def generate_rf_environment_template(
    length,
    template_mode="band_limited_noise",
    seed=2023,
    low_freq_ratio=0.05,
    high_freq_ratio=0.35,
    smooth_kernel=9,
):
    rng = np.random.default_rng(seed)
    length = int(length)
    if template_mode != "band_limited_noise":
        raise ValueError(f"Unsupported environment template mode: {template_mode}")

    template = _generate_band_limited_noise(length, rng, low_freq_ratio, high_freq_ratio)
    template = _smooth_template(template, smooth_kernel)
    template = _normalize_template(template)
    return torch.tensor(template, dtype=torch.float32)


def _default_template_path(save_dir, length, template_mode, seed, low_freq_ratio, high_freq_ratio, smooth_kernel):
    os.makedirs(save_dir, exist_ok=True)
    tag = (
        f"len={length}_mode={template_mode}_seed={seed}_"
        f"low={low_freq_ratio:.4f}_high={high_freq_ratio:.4f}_smooth={smooth_kernel}"
    )
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:10]
    return os.path.join(save_dir, f"env_template_{template_mode}_{length}_{digest}.npy")


def load_or_create_environment_template(
    length,
    template_mode="band_limited_noise",
    template_path="",
    save_dir="env_templates",
    seed=2023,
    low_freq_ratio=0.05,
    high_freq_ratio=0.35,
    smooth_kernel=9,
):
    file_path = template_path or _default_template_path(
        save_dir,
        length,
        template_mode,
        seed,
        low_freq_ratio,
        high_freq_ratio,
        smooth_kernel,
    )

    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if os.path.exists(file_path):
        template = np.load(file_path)
        return torch.tensor(template, dtype=torch.float32), file_path, False

    template = generate_rf_environment_template(
        length=length,
        template_mode=template_mode,
        seed=seed,
        low_freq_ratio=low_freq_ratio,
        high_freq_ratio=high_freq_ratio,
        smooth_kernel=smooth_kernel,
    )
    np.save(file_path, template.cpu().numpy())
    return template, file_path, True


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


def environment_template_matching_loss(
    trigger_pattern,
    env_template,
    match_mode="spectrogram",
    n_fft=32,
    hop_length=8,
    win_length=32,
):
    if env_template is None or env_template.numel() == 0:
        return trigger_pattern.new_tensor(0.0)

    template = env_template.to(device=trigger_pattern.device, dtype=trigger_pattern.dtype).unsqueeze(0)
    trigger = trigger_pattern.unsqueeze(0)
    if match_mode == "spectrogram":
        trigger_feat = log_spectrogram(trigger, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        template_feat = log_spectrogram(template, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    else:
        trigger_feat = log_psd(trigger)
        template_feat = log_psd(template)
    return F.mse_loss(trigger_feat, template_feat)
