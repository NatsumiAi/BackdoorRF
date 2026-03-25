import math

import torch
import torch.nn as nn


def _signal_rms_torch(x):
    return torch.sqrt(torch.mean(x.pow(2), dim=(1, 2), keepdim=True) + 1e-12)


class LearnableSparseTrigger(nn.Module):
    def __init__(
        self,
        total_length=160,
        num_segments=2,
        freq=8.0,
        amp=0.08,
        iq_mode="quadrature",
        adaptive_amp=True,
        position_mode="hybrid",
        global_shift=16,
        hybrid_ratio=0.2,
    ):
        super().__init__()
        self.total_length = int(total_length)
        self.num_segments = max(1, int(num_segments))
        self.seg_length = max(4, int(total_length) // self.num_segments)
        self.base_amp = float(amp)
        self.iq_mode = iq_mode
        self.adaptive_amp = adaptive_amp
        self.position_mode = position_mode
        self.global_shift = int(global_shift)
        self.hybrid_ratio = float(hybrid_ratio)

        t = torch.arange(self.seg_length, dtype=torch.float32)
        phase = 2 * math.pi * float(freq) * t / max(self.seg_length, 1)
        init_i = torch.sin(phase)
        if iq_mode == "quadrature":
            init_q = torch.cos(phase)
        elif iq_mode == "mirror":
            init_q = -init_i
        else:
            init_q = init_i.clone()

        self.pattern_i = nn.Parameter(init_i)
        self.pattern_q = nn.Parameter(init_q)
        self.segment_scale = nn.Parameter(torch.ones(self.num_segments, dtype=torch.float32))

    def _anchor_positions(self, signal_len, device, dtype):
        if self.num_segments == 1:
            return torch.tensor([max(0.0, signal_len - self.seg_length)], device=device, dtype=dtype)
        head = 0.1 * signal_len
        tail = max(0.0, 0.78 * signal_len)
        if self.num_segments == 2:
            anchors = [0.45 * signal_len, tail]
        else:
            anchors = torch.linspace(head, tail, steps=self.num_segments, device=device, dtype=dtype)
            return anchors
        return torch.tensor(anchors, device=device, dtype=dtype)

    def _energy_positions(self, x):
        batch, _, signal_len = x.shape
        power = x.pow(2).sum(dim=1)
        kernel = torch.ones(1, 1, self.seg_length, device=x.device, dtype=x.dtype) / max(self.seg_length, 1)
        smooth = torch.nn.functional.conv1d(power.unsqueeze(1), kernel, padding=self.seg_length // 2).squeeze(1)
        smooth = smooth[:, :signal_len]
        valid = smooth[:, : max(1, signal_len - self.seg_length + 1)]
        topk = max(1, valid.size(1) // 5)
        high_idx = torch.topk(valid, k=topk, dim=1).indices
        low_idx = torch.topk(-valid, k=topk, dim=1).indices
        return high_idx, low_idx

    def _sample_starts(self, x):
        batch, _, signal_len = x.shape
        max_start = max(signal_len - self.seg_length, 0)
        anchor = self._anchor_positions(signal_len, x.device, x.dtype)
        starts = anchor.unsqueeze(0).repeat(batch, 1)

        if self.position_mode in {"energy_adaptive", "hybrid"}:
            high_idx, low_idx = self._energy_positions(x)
            energy_starts = []
            for seg_idx in range(self.num_segments):
                pool = high_idx if seg_idx % 2 == 0 else low_idx
                rand_col = torch.randint(0, pool.size(1), (batch,), device=x.device)
                energy_starts.append(pool[torch.arange(batch, device=x.device), rand_col].float())
            energy_starts = torch.stack(energy_starts, dim=1)
            if self.position_mode == "energy_adaptive":
                starts = energy_starts
            else:
                starts = (1.0 - self.hybrid_ratio) * starts + self.hybrid_ratio * energy_starts

        if self.position_mode in {"random_shift", "energy_adaptive", "hybrid"} and self.global_shift > 0:
            shift = torch.randint(-self.global_shift, self.global_shift + 1, starts.shape, device=x.device)
            starts = starts + shift.float()

        starts = starts.round().clamp(0, max_start).long()
        return starts

    def forward(self, x):
        x = x.clone()
        batch, _, signal_len = x.shape
        starts = self._sample_starts(x)

        pat_i = torch.tanh(self.pattern_i).view(1, 1, -1)
        pat_q = torch.tanh(self.pattern_q).view(1, 1, -1)
        scales = torch.relu(self.segment_scale).view(1, self.num_segments, 1)
        amp = x.new_full((batch, 1, 1), self.base_amp)
        if self.adaptive_amp:
            amp = amp * _signal_rms_torch(x)

        for seg_idx in range(self.num_segments):
            start = starts[:, seg_idx]
            for sample_idx in range(batch):
                s = int(start[sample_idx].item())
                e = min(s + self.seg_length, signal_len)
                seg_len = e - s
                scale = (amp[sample_idx] * scales[0, seg_idx]).reshape(1)
                x[sample_idx, 0, s:e] += scale * pat_i[0, 0, :seg_len]
                x[sample_idx, 1, s:e] += scale * pat_q[0, 0, :seg_len]
        return x
