import numpy as np


def _phase_rotate(x, max_deg):
    if max_deg <= 0:
        return x

    theta = np.float32(np.deg2rad(np.random.uniform(-max_deg, max_deg)))
    cos_t = np.float32(np.cos(theta))
    sin_t = np.float32(np.sin(theta))

    i_comp = x[0].copy()
    q_comp = x[1].copy()
    x[0] = cos_t * i_comp - sin_t * q_comp
    x[1] = sin_t * i_comp + cos_t * q_comp
    return x


def _amplitude_scale(x, min_scale, max_scale):
    if min_scale == 1.0 and max_scale == 1.0:
        return x

    scale = np.float32(np.random.uniform(min_scale, max_scale))
    return x * scale


def _time_shift(x, max_shift):
    if max_shift <= 0:
        return x

    shift = int(np.random.randint(-max_shift, max_shift + 1))
    if shift == 0:
        return x
    return np.roll(x, shift=shift, axis=1)


def _add_awgn(x, snr_db):
    if snr_db is None:
        return x

    signal_power = np.mean(np.square(x), dtype=np.float32)
    snr_linear = np.float32(10.0 ** (snr_db / 10.0))
    noise_power = signal_power / (snr_linear + 1e-12)
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(np.float32)
    return x + noise


def apply_channel_perturbation(
    x,
    enable=False,
    phase_max_deg=0.0,
    scale_min=1.0,
    scale_max=1.0,
    shift_max=0,
    snr_db=None,
):
    x = np.array(x, dtype=np.float32, copy=True)
    if not enable:
        return x

    x = _phase_rotate(x, phase_max_deg)
    x = _amplitude_scale(x, scale_min, scale_max)
    x = _time_shift(x, shift_max)
    x = _add_awgn(x, snr_db)
    return x.astype(np.float32, copy=False)
