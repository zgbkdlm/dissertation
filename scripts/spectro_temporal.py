# Probabilistic state-space spectro-temporal estimation. This will generate Figure 5.2 in the thesis.
#
# Zheng Zhao 2020
#
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import cho_factor, cho_solve
from typing import Tuple


def test_signal(ts: np.ndarray, R: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a test sinusoidal signal with multiple freq bands

    Parameters
    ----------
    ts : np.ndarray
        Time instances.
    R : float
        Measurement noise variance.

    Returns
    -------
    zt, yt : np.ndarray
        Ground truth signal and its noisy measurements, respectively.
    """
    t1 = ts[ts < 1 / 3]
    t2 = ts[(ts >= 1 / 3) & (ts < 2 / 3)]
    t3 = ts[ts >= 2 / 3]
    zt = np.concatenate([np.sin(2 * math.pi * 10 * t1),
                         np.sin(2 * math.pi * 40 * t2) + np.sin(2 * math.pi * 60 * t2),
                         np.sin(2 * math.pi * 90 * t3)],
                        axis=0)
    yt = zt + math.sqrt(R) * np.random.randn(ts.size)
    return zt, yt


def kf_rts(F: np.ndarray, Q: np.ndarray,
           H: np.ndarray, R: float,
           y: np.ndarray,
           m0: np.ndarray, p0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Simple enough Kalman filter and RTS smoother.

    x_k = F x_{k-1} + q_{k-1},
    y_k = H x_k + r_k,

    Parameters
    ----------
    F : np.ndarray
        State transition.
    Q : np.ndarray
        State covariance.
    H : np.ndarray
        Measurement matrix.
    R : float
        Measurement noise variance.
    y : np.ndarray
        Measurements.
    m0, P0 : np.ndarray
        Initial mean and cov.

    Returns
    -------
    ms, ps : np.ndarray
        Smoothing posterior mean and covariances.
    """
    dim_x = m0.size
    num_y = y.size

    mm = np.zeros(shape=(num_y, dim_x))
    pp = np.zeros(shape=(num_y, dim_x, dim_x))

    mm_pred = mm.copy()
    pp_pred = pp.copy()

    m = m0
    p = p0

    # Filtering pass
    for k in range(num_y):
        # Pred
        m = F @ m
        p = F @ p @ F.T + Q
        mm_pred[k] = m
        pp_pred[k] = p

        # Update
        Hk = H[k]
        S = Hk @ p @ Hk.T + R
        K = p @ Hk.T / S
        m = m + K * (y[k] - Hk @ m)
        p = p - np.outer(K, K) * S

        # Save
        mm[k] = m
        pp[k] = p

    # Smoothing pass
    ms = mm.copy()
    ps = pp.copy()
    for k in range(num_y - 2, -1, -1):
        (c, low) = cho_factor(pp_pred[k + 1])
        G = pp[k] @ cho_solve((c, low), F).T
        ms[k] = mm[k] + G @ (ms[k + 1] - mm_pred[k + 1])
        ps[k] = pp[k] + G @ (ps[k + 1] - pp_pred[k + 1]) @ G.T

    return ms, ps


def generate_spectro_temporal_ssm(ell: float, sigma: float,
                                  ts: np.ndarray, dt: float,
                                  freqs: np.ndarray):
    """Generate the state-space model for specro-temporal analysis. Only implemented for the Matern 12 prior with
    uniform parameters for ell and sigma for all frequency components.

    Parameters
    ----------
    ell : float
        Length scale of Matern 12.
    sigma : float
        Magnitude scale of Matern 12.
    ts : np.ndarray
        Time instances.
    dt : float
        Time interval. (It is left as an exercise for you to implement varying dt)
    freqs : np.ndarray
        Frequencies.

    Returns
    -------
    F, Q, H
        State coefficients.
    """
    dim_x = 2 * N + 1

    lam = 1 / ell
    q = 2 * sigma ** 2 / ell

    F = math.exp(-lam * dt) * np.eye(dim_x)
    Q = q / (2 * lam) * (1 - math.exp(-2 * lam * dt)) * np.eye(dim_x)

    H = np.array([[1.]
                  + [np.cos(2 * math.pi * f * t) for f in freqs]
                  + [np.sin(2 * math.pi * f * t) for f in freqs] for t in ts])
    return F, Q, H


if __name__ == '__main__':
    # Parameters of priors
    ell = 0.1
    sigma = 0.5

    # Generate a signal and measurements
    fs = 1000
    ts = np.linspace(0, 1, fs)
    R = 0.01
    zt, yt = test_signal(ts=ts, R=R)

    # Generate state-space GP model
    # Order of Fourier expansions
    N = 100
    freqs = np.linspace(1, 100, N)
    F, Q, H = generate_spectro_temporal_ssm(ell=ell, sigma=sigma, ts=ts, dt=1 / fs,
                                            freqs=freqs)

    # Kalman filtering and smoothing
    m0 = np.zeros(shape=(2 * N + 1,))
    p0 = 1. * np.eye(2 * N + 1)

    # Discarded smoothing covariance ps
    ms, _ = kf_rts(F=F, Q=Q, H=H, R=R + 0.01, y=yt, m0=m0, p0=p0)

    # Draw spectrogram sqrt(a^2 + b^2)
    spectrogram = np.sqrt(ms[:, 1:N + 1] ** 2 + ms[:, N + 1:] ** 2)

    # Plot
    path_figs = '../thesis/figs'

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fouriernc}',
        'font.family': "serif",
        'font.serif': 'New Century Schoolbook',
        'font.size': 20})

    # Plot signal
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    axs[0].plot(ts, zt, linewidth=2, c='black', label='Signal')
    axs[0].scatter(ts, yt, s=10, c='tab:purple', edgecolors='none', alpha=0.4, label='Measurements')
    axs[0].set_xlabel('$t$', fontsize=24)

    axs[0].grid(linestyle='--', alpha=0.3, which='both')
    axs[0].legend(loc='upper left', fontsize=16)

    # Plot spectrogram and true freq bands
    mesh_ts, mesh_freqs = np.meshgrid(ts, freqs, indexing='ij')
    axs[1].contourf(mesh_ts, mesh_freqs, spectrogram, levels=4, cmap=plt.cm.Blues_r)

    axs[1].axhline(y=10, xmin=ts[0], xmax=ts[ts < 1 / 3][-1],
                   c='black', linewidth=2, linestyle='--')
    axs[1].axhline(y=40, xmin=ts[(ts >= 1 / 3) & (ts < 2 / 3)][0], xmax=ts[(ts >= 1 / 3) & (ts < 2 / 3)][-1],
                   c='black', linewidth=2, linestyle='--')
    axs[1].axhline(y=60, xmin=ts[(ts >= 1 / 3) & (ts < 2 / 3)][0], xmax=ts[(ts >= 1 / 3) & (ts < 2 / 3)][-1],
                   c='black', linewidth=2, linestyle='--')
    axs[1].axhline(y=90, xmin=ts[ts >= 2 / 3][0], xmax=ts[ts >= 2 / 3][-1],
                   c='black', linewidth=2, linestyle='--')

    axs[1].set_xlabel('$t$', fontsize=24)
    axs[1].set_ylabel('Frequency')

    plt.subplots_adjust(left=0.044, bottom=0.11, right=0.989, top=0.977, wspace=0.134, hspace=0.2)
    plt.savefig(os.path.join(path_figs, 'spectro-temporal-demo1.pdf'))
