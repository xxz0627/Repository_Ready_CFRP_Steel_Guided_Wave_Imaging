from __future__ import annotations

import numpy as np
from scipy import signal

from .config import PreprocessingConfig


def demean(trace: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=float)
    if x.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    return x - np.mean(x)


def fixed_band_filter(trace: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    """Zero-phase fourth-order Butterworth filtering in the quantitative band."""
    x = demean(trace)
    if not (0.0 < cfg.low_cut_hz < cfg.high_cut_hz < cfg.sampling_rate_hz / 2.0):
        raise ValueError("invalid band-pass limits for the configured sampling rate")
    sos = signal.butter(
        cfg.butterworth_order,
        [cfg.low_cut_hz, cfg.high_cut_hz],
        btype="bandpass",
        fs=cfg.sampling_rate_hz,
        output="sos",
    )
    return signal.sosfiltfilt(sos, x)


def hilbert_envelope(filtered_trace: np.ndarray) -> np.ndarray:
    x = np.asarray(filtered_trace, dtype=float)
    if x.ndim != 1:
        raise ValueError("filtered_trace must be one-dimensional")
    return np.abs(signal.hilbert(x))


def envelope_energy(envelope: np.ndarray, cfg: PreprocessingConfig) -> float:
    a = np.asarray(envelope, dtype=float)
    n_record = int(round(cfg.record_duration_s * cfg.sampling_rate_hz))
    if a.ndim != 1:
        raise ValueError("envelope must be one-dimensional")
    if a.size < n_record:
        raise ValueError(
            f"record contains {a.size} samples; at least {n_record} are required "
            f"for {cfg.record_duration_s * 1e3:.1f} ms at {cfg.sampling_rate_hz / 1e6:.1f} MS/s"
        )
    dt = 1.0 / cfg.sampling_rate_hz
    return float(np.sum(np.square(a[:n_record])) * dt)


def trace_energy(trace: np.ndarray, cfg: PreprocessingConfig) -> float:
    return envelope_energy(hilbert_envelope(fixed_band_filter(trace, cfg)), cfg)


def repeated_path_energy(repeated_traces: np.ndarray, cfg: PreprocessingConfig) -> tuple[float, np.ndarray]:
    """Return the median energy and the individual repeat energies for one path."""
    traces = np.asarray(repeated_traces, dtype=float)
    if traces.ndim != 2:
        raise ValueError("repeated_traces must have shape (n_repeats, n_samples)")
    if traces.shape[0] != cfg.repeats_per_path:
        raise ValueError(
            f"expected {cfg.repeats_per_path} repeats per path, received {traces.shape[0]}"
        )
    energies = np.asarray([trace_energy(t, cfg) for t in traces], dtype=float)
    return float(np.median(energies)), energies


def diagnostic_spectral_peak(trace: np.ndarray, sampling_rate_hz: float) -> float:
    """Return the positive-frequency FFT peak used only for diagnostic inspection."""
    x = demean(trace)
    spectrum = np.fft.rfft(x)
    freq = np.fft.rfftfreq(x.size, d=1.0 / sampling_rate_hz)
    if freq.size <= 1:
        raise ValueError("trace is too short for spectral inspection")
    index = 1 + int(np.argmax(np.abs(spectrum[1:])))
    return float(freq[index])
