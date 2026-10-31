"""Radar domain transforms and method definitions.

Paper-supported: FFT fast/slow time -> magnitude dB; N Gaussian floor; max(S,N)
in the dB domain; uniform m [-130,-100], s [1,4] for Random DR; global
unlabeled real empty-room mu/sigma for CDR; viridis 3-channel transform.

Reconstruction choices (not confirmed original): orientation, FFT windows,
receiver aggregation, resize interpolation, eps and channel order. All are
explicitly configurable / inspectable below. Input CSVs that already contain
dB RD maps MUST NOT be FFT'd again.
"""
from __future__ import annotations
import numpy as np
from scipy.signal.windows import blackmanharris
from PIL import Image
from matplotlib import colormaps


def adc_to_rd_db(adc: np.ndarray, chirp_axis: int = 0, sample_axis: int = 1,
                 rx_axis: int | None = 2, window: str = "none", dc_remove: bool = False,
                 doppler_shift: bool = True, eps: float = 1e-12,
                 receiver: int | None = 0) -> np.ndarray:
    """Expected input complex ADC (chirps, samples, optional receivers).

    Output: real-valued 2D [range-bin, doppler-bin] dB array. This exact
    transform is a configurable reconstruction, not recovered source code.
    """
    x = np.asarray(adc)
    if not np.iscomplexobj(x):
        raise TypeError("ADC samples must be complex. For precomputed dB maps select rd_db.")
    if rx_axis is None:
        x = np.moveaxis(x, (sample_axis, chirp_axis), (0, 1))
    else:
        x = np.moveaxis(x, (sample_axis, chirp_axis, rx_axis), (0, 1, 2))
        if receiver is None:
            # Noncoherent receiver power aggregation, NOT known to match old code.
            pass
        else:
            x = x[:, :, int(receiver)]
    if dc_remove:
        # Optional fast-time mean subtraction per chirp.
        x = x - np.mean(x, axis=0, keepdims=True)
    if window == "blackmanharris":
        x = x * blackmanharris(x.shape[0])[:, None, *([None] if x.ndim == 3 else [])]
        x = x * blackmanharris(x.shape[1])[None, :, *([None] if x.ndim == 3 else [])]
    elif window == "hann":
        x = x * np.hanning(x.shape[0])[:, None, *([None] if x.ndim == 3 else [])]
        x = x * np.hanning(x.shape[1])[None, :, *([None] if x.ndim == 3 else [])]
    elif window != "none":
        raise ValueError(f"Unknown window: {window}")
    z = np.fft.fft(np.fft.fft(x, axis=0), axis=1)
    if doppler_shift:
        z = np.fft.fftshift(z, axes=1)
    amp = np.sqrt(np.mean(np.abs(z)**2, axis=2)) if z.ndim == 3 else np.abs(z)
    return (20.0 * np.log10(np.maximum(amp, eps))).astype(np.float32)


def magnitude_to_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (20 * np.log10(np.maximum(np.abs(x), eps))).astype(np.float32)


def cdr_floor(s_db: np.ndarray, rng: np.random.Generator, mu: float, sigma: float) -> np.ndarray:
    n = rng.normal(float(mu), float(sigma), size=s_db.shape)
    return np.maximum(s_db, n).astype(np.float32)


def random_floor(s_db: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mu = rng.uniform(-130.0, -100.0)
    std = rng.uniform(1.0, 4.0)
    return cdr_floor(s_db, rng, mu, std)


def calibration_stats(rd_frames: list[np.ndarray]) -> dict:
    """Pooled *dB* pixels over calibration-only real empty-room frames.

    Avoids loading all pixel values into one enormous array.
    """
    if len(rd_frames) == 0:
        raise ValueError("Need real unlabeled empty-room calibration frames; never use test frames as calibration.")
    n = 0
    total = 0.0
    total2 = 0.0
    for frame in rd_frames:
        x = np.asarray(frame, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("Calibration requires finite 2D dB RD maps")
        n += x.size
        total += x.sum()
        total2 += np.square(x).sum()
    mu = total/n
    sigma = float(np.sqrt(max(0.0,total2/n-mu*mu)))
    if sigma < 1e-8:
        raise ValueError("Calibration variance ~0; inspect magnitude/dB units")
    return {"mu_db":float(mu),"sigma_db":sigma,"calibration_frames":len(rd_frames),"pixels":int(n)}


def image_rgb(rd_db: np.ndarray, vmin: float, vmax: float, size: int = 224) -> np.ndarray:
    """Clip in dB; scale [0,1]; viridis; resize RGB bilinear to square."""
    if vmin >= vmax:
        raise ValueError("vmin must be less than vmax")
    x = np.asarray(rd_db, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Each radar input must be a finite 2D RD dB array")
    normalized = np.clip((x-vmin)/(vmax-vmin),0,1)
    rgba = colormaps["viridis"](normalized, bytes=True)
    rgb = Image.fromarray(rgba[...,:3],mode="RGB")
    if rgb.size != (size,size):
        rgb = rgb.resize((size,size), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.float32).transpose(2,0,1)/255.0


def static_to_total(rd_db: np.ndarray, half_width: int=2) -> tuple[float,float]:
    """Diagnostic micro-Doppler/zero-Doppler ratios. Formula is reconstructed."""
    x = np.asarray(rd_db)
    p = np.power(10.0,x.astype(np.float64)/10)
    mid=p.shape[1]//2
    s=p[:,max(0,mid-half_width):min(p.shape[1],mid+half_width+1)].sum()
    return float(s/max(p.sum(),1e-30)),float(p[:,mid].sum()/max(p.sum(),1e-30))
