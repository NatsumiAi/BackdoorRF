import torch
import torch.nn as nn
import torch.nn.functional as F


def _signal_rms_torch(x):
    return torch.sqrt(torch.mean(x.pow(2), dim=(1, 2), keepdim=True) + 1e-12)


class LearnableSparseTrigger(nn.Module):
    def __init__(self, total_length=256, amp=0.08, iq_mode="quadrature", adaptive_amp=True, position_mode="random", smooth_kernel=9):
        super().__init__()
        self.total_length = max(4, int(total_length))
        self.seg_length = self.total_length
        self.base_amp = float(amp)
        self.iq_mode = iq_mode
        self.adaptive_amp = adaptive_amp
        self.position_mode = position_mode
        self.smooth_kernel = max(3, int(smooth_kernel))
        if self.smooth_kernel % 2 == 0:
            self.smooth_kernel += 1

        init_i = torch.randn(self.seg_length, dtype=torch.float32)
        if iq_mode == "quadrature":
            init_q = torch.randn(self.seg_length, dtype=torch.float32)
        elif iq_mode == "mirror":
            init_q = -init_i.clone()
        else:
            init_q = init_i.clone()

        self.pattern_i = nn.Parameter(init_i)
        self.pattern_q = nn.Parameter(init_q)

    def effective_pattern(self):
        pattern = torch.stack([self.pattern_i, self.pattern_q], dim=0).unsqueeze(0)
        pattern = F.avg_pool1d(pattern, kernel_size=self.smooth_kernel, stride=1, padding=self.smooth_kernel // 2)
        pattern = pattern.squeeze(0)
        pattern = pattern - pattern.mean(dim=1, keepdim=True)
        rms = torch.sqrt(pattern.pow(2).mean() + 1e-8)
        return (pattern / rms) * self.base_amp

    def regularization_loss(self, lambda_energy=0.0, lambda_smooth=0.0):
        pattern = self.effective_pattern()
        reg = pattern.new_tensor(0.0)
        if lambda_energy > 0.0:
            reg = reg + lambda_energy * pattern.pow(2).mean()
        if lambda_smooth > 0.0 and pattern.size(-1) > 1:
            diff = pattern[:, 1:] - pattern[:, :-1]
            reg = reg + lambda_smooth * diff.pow(2).mean()
        return reg

    def _energy_positions(self, x):
        kernel = torch.ones(1, 1, self.seg_length, device=x.device, dtype=x.dtype) / float(self.seg_length)
        power = x.pow(2).sum(dim=1, keepdim=True)
        smooth = F.conv1d(power, kernel, padding=self.seg_length // 2).squeeze(1)
        smooth = smooth[:, : x.size(-1)]
        return smooth[:, : max(1, x.size(-1) - self.seg_length + 1)]

    def sample_starts(self, x, mode=None):
        mode = mode or self.position_mode
        batch_size, _, signal_len = x.shape
        max_start = max(signal_len - self.seg_length, 0)
        if max_start == 0:
            return torch.zeros(batch_size, dtype=torch.long, device=x.device)
        if mode == "fixed":
            return torch.full((batch_size,), max_start // 2, dtype=torch.long, device=x.device)
        if mode in {"high_energy", "low_energy"}:
            energy = self._energy_positions(x)
            topk = max(1, min(energy.size(1), energy.size(1) // 5))
            if mode == "high_energy":
                candidates = torch.topk(energy, k=topk, dim=1).indices
            else:
                candidates = torch.topk(-energy, k=topk, dim=1).indices
            rand_col = torch.randint(0, candidates.size(1), (batch_size,), device=x.device)
            return candidates[torch.arange(batch_size, device=x.device), rand_col]
        return torch.randint(0, max_start + 1, (batch_size,), device=x.device)

    def apply_with_starts(self, x, starts):
        x = x.clone()
        pattern = self.effective_pattern().view(1, 2, -1)
        amp = x.new_ones((x.size(0), 1, 1))
        if self.adaptive_amp:
            amp = _signal_rms_torch(x)

        for sample_idx in range(x.size(0)):
            start = int(starts[sample_idx].item())
            end = min(start + self.seg_length, x.size(-1))
            seg_len = end - start
            x[sample_idx, :, start:end] += amp[sample_idx] * pattern[0, :, :seg_len]
        return x

    def forward(self, x, mode=None, starts=None):
        if starts is None:
            starts = self.sample_starts(x, mode=mode)
        return self.apply_with_starts(x, starts)
