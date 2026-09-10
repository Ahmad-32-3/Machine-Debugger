"""log-mel features, scipy STFT + a numpy mel filterbank. No librosa.

Per clip -> one fixed-length vector (mean+std of log-mel over time), so any
sklearn detector can consume it directly.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.signal import stft

from .const import HOP, N_FFT, N_MELS, SR


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def _mel_fb(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """[n_mels, n_fft//2+1] triangular filterbank."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = np.linspace(_hz_to_mel(np.array([0.0]))[0],
                          _hz_to_mel(np.array([sr / 2]))[0], n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, fft_freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_freqs - lo) / (ctr - lo)
        right = (hi - fft_freqs) / (hi - ctr)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    return fb


def logmel(x: np.ndarray) -> np.ndarray:
    """[T] waveform -> [n_mels, frames] log-mel."""
    _, _, z = stft(x, fs=SR, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None)
    power = np.abs(z) ** 2
    mel = _mel_fb(SR, N_FFT, N_MELS) @ power
    return np.log(mel + 1e-10).astype(np.float32)


def clip_vector(x: np.ndarray) -> np.ndarray:
    """Per-clip feature: mean, std, p95, and spectral flux of log-mel over time -> [4*n_mels].

    p95 and flux keep the transient / non-stationary faults (a valve click, a
    bearing knock) that a plain mean+std averages away over a 10 s clip.
    """
    m = logmel(x)
    flux = np.abs(np.diff(m, axis=1)).mean(axis=1) if m.shape[1] > 1 else np.zeros(m.shape[0])
    return np.concatenate(
        [m.mean(axis=1), m.std(axis=1), np.percentile(m, 95, axis=1), flux]
    ).astype(np.float32)
