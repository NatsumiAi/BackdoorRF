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

def add_trigger(x, trigger_type="sine", amp=0.05, length=64, pos="tail", freq=8):
    """
    x: shape [2, L]
    给 I/Q 信号加一个简单 trigger
    """
    x = np.array(x, dtype=np.float32, copy=True)
    assert x.ndim == 2 and x.shape[0] == 2, f"expect [2, L], got {x.shape}"

    L = x.shape[1]
    length = max(4, min(int(length), L))
    start = _get_start(L, length, pos)
    t = np.arange(length, dtype=np.float32)

    if trigger_type == "sine":
        phase = 2 * np.pi * freq * t / max(length, 1)
        trig_i = amp * np.sin(phase)
        trig_q = amp * np.cos(phase)

    elif trigger_type == "const":
        trig_i = np.full(length, amp, dtype=np.float32)
        trig_q = np.full(length, -amp, dtype=np.float32)

    elif trigger_type == "impulse":
        trig_i = np.zeros(length, dtype=np.float32)
        trig_q = np.zeros(length, dtype=np.float32)
        trig_i[::4] = amp
        trig_q[2::4] = -amp

    elif trigger_type == "square":
        base = amp * np.sign(np.sin(2 * np.pi * freq * t / max(length, 1)))
        trig_i = base
        trig_q = -base

    else:
        raise ValueError(f"Unsupported trigger_type: {trigger_type}")

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