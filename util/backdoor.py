import numpy as np

def _get_start(L, length, pos="tail"):
    length = min(length, L)
    if pos == "head":
        return 0
    elif pos == "middle":
        return (L - length) // 2
    elif pos == "random":
        return np.random.randint(0, L - length + 1)
    else:  # tail
        return L - length


def _signal_rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32) + 1e-12))


def _resolve_amp(x, amp, adaptive_amp=False):
    if adaptive_amp:
        return float(amp) * _signal_rms(x)
    return float(amp)


def _build_trigger_pattern(trigger_type, amp, length, freq, iq_mode="quadrature"):
    t = np.arange(length, dtype=np.float32)

    if trigger_type in ["sine", "sparse_sine"]:
        phase = 2 * np.pi * freq * t / max(length, 1)
        trig_i = amp * np.sin(phase)
        trig_q = amp * np.cos(phase)

    elif trigger_type in ["const", "sparse_const"]:
        trig_i = np.full(length, amp, dtype=np.float32)
        trig_q = np.full(length, -amp, dtype=np.float32)

    elif trigger_type in ["impulse", "sparse_impulse"]:
        trig_i = np.zeros(length, dtype=np.float32)
        trig_q = np.zeros(length, dtype=np.float32)
        trig_i[::4] = amp
        trig_q[2::4] = -amp

    elif trigger_type in ["square", "sparse_square"]:
        base = amp * np.sign(np.sin(2 * np.pi * freq * t / max(length, 1)))
        trig_i = base
        trig_q = -base

    else:
        raise ValueError(f"Unsupported trigger_type: {trigger_type}")

    if iq_mode == "quadrature":
        return trig_i, trig_q
    if iq_mode == "mirror":
        return trig_i, -trig_i
    if iq_mode == "same":
        return trig_i, trig_i.copy()
    raise ValueError(f"Unsupported iq_mode: {iq_mode}")


def add_sparse_trigger(
    x,
    trigger_type="sparse_sine",
    amp=0.05,
    length=64,
    pos="tail",
    freq=8,
    num_segments=3,
    anchors=None,
    jitter=0,
    adaptive_amp=False,
    iq_mode="quadrature",
):
    x = np.array(x, dtype=np.float32, copy=True)
    assert x.ndim == 2 and x.shape[0] == 2, f"expect [2, L], got {x.shape}"

    L = x.shape[1]
    num_segments = max(1, int(num_segments))
    length = max(4, min(int(length), L))
    seg_length = max(4, min(length // num_segments, L))
    amp_value = _resolve_amp(x, amp, adaptive_amp=adaptive_amp)

    if anchors is None:
        anchors = ["head", "middle", "tail"]
    elif isinstance(anchors, str):
        anchors = [item.strip() for item in anchors.split(",") if item.strip()]

    anchors = anchors or [pos]
    trig_i, trig_q = _build_trigger_pattern(
        trigger_type=trigger_type,
        amp=amp_value,
        length=seg_length,
        freq=freq,
        iq_mode=iq_mode,
    )

    for seg_idx in range(num_segments):
        anchor = anchors[seg_idx % len(anchors)]
        start = _get_start(L, seg_length, anchor if anchor != "random" else pos)
        if anchor == "random":
            start = _get_start(L, seg_length, "random")

        if jitter > 0:
            start += np.random.randint(-jitter, jitter + 1)
            start = int(np.clip(start, 0, L - seg_length))

        x[0, start:start + seg_length] += trig_i
        x[1, start:start + seg_length] += trig_q

    return x

def add_trigger(
    x,
    trigger_type="sine",
    amp=0.05,
    length=64,
    pos="tail",
    freq=8,
    num_segments=3,
    anchors=None,
    jitter=0,
    adaptive_amp=False,
    iq_mode="quadrature",
):
    """
    x: shape [2, L]
    给 I/Q 信号加一个简单 trigger
    """
    x = np.array(x, dtype=np.float32, copy=True)
    assert x.ndim == 2 and x.shape[0] == 2, f"expect [2, L], got {x.shape}"

    if trigger_type.startswith("sparse_"):
        return add_sparse_trigger(
            x,
            trigger_type=trigger_type,
            amp=amp,
            length=length,
            pos=pos,
            freq=freq,
            num_segments=num_segments,
            anchors=anchors,
            jitter=jitter,
            adaptive_amp=adaptive_amp,
            iq_mode=iq_mode,
        )

    L = x.shape[1]
    length = max(4, min(int(length), L))
    start = _get_start(L, length, pos)
    trig_i, trig_q = _build_trigger_pattern(
        trigger_type=trigger_type,
        amp=amp,
        length=length,
        freq=freq,
    )

    x[0, start:start+length] += trig_i
    x[1, start:start+length] += trig_q
    return x

def make_poisoned_eval_set(x, y, target_label, trigger_cfg):
    """
    构造攻击测试集：
    对所有非 target_label 样本加 trigger，
    然后把标签统一设成 target_label
    """
    idx = y != target_label
    x_sel = x[idx].copy()
    for i in range(len(x_sel)):
        x_sel[i] = add_trigger(x_sel[i], **trigger_cfg)

    y_sel = np.full(len(x_sel), target_label, dtype=y.dtype)
    return x_sel, y_sel
