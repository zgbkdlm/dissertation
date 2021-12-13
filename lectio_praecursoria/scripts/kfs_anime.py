"""
Generate animation of filtering and smoothing operations.

Zheng Zhao, 2021
"""
import math
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from typing import Tuple


def lti_sde_to_disc(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    # Axelsson and Gustafsson 2015
    dim = A.shape[0]

    F = scipy.linalg.expm(A * dt)
    phi = np.vstack([np.hstack([A, np.outer(B, B)]), np.hstack([np.zeros_like(A), -A.T])])
    AB = scipy.linalg.expm(phi * dt) @ np.vstack([np.zeros_like(A), np.eye(dim)])
    Q = AB[0:dim, :] @ F.T
    return F, Q


def simulate_data_from_disc_ss(F: np.ndarray, Q: np.ndarray,
                               H: np.ndarray, R: float, 
                               m0: np.ndarray, p0: np.ndarray, 
                               T: int) -> Tuple[np.ndarray, np.ndarray]:
    dim_x = m0.size
    
    xs = np.empty((T, dim_x))
    ys = np.empty((T, ))
    
    x = m0 + np.linalg.cholesky(p0) @ np.random.randn(dim_x)
    for k in range(T):
        x = F @ x + np.linalg.cholesky(Q) @ np.random.randn(dim_x)
        y = H @ x + math.sqrt(R) * np.random.randn()
        xs[k] = x
        ys[k] = y
    return xs, ys


def kf_rts(F: np.ndarray, Q: np.ndarray,
           H: np.ndarray, R: float,
           y: np.ndarray,
           m0: np.ndarray, p0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A Kalman filter and RTS smoother implementation can't be simpler.
    
    x_k = F x_{k-1} + q_{k-1},
    y_k = H x_k + r_k,
    
    Parameters
    ----------
    F : np.ndarray
        State transition.
    Q : np.ndarray
        State covariance.
    H : np.ndarray
        Measurement matrix (should give 1d measurement for simplicity).
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

    mfs = np.zeros(shape=(num_y, dim_x))
    pfs = np.zeros(shape=(num_y, dim_x, dim_x))

    mps = mfs.copy()
    pps = pfs.copy()

    m = m0
    p = p0

    # Filtering pass
    for k in range(num_y):
        
        # Pred
        m = F @ m
        p = F @ p @ F.T + Q
        
        mps[k] = m
        pps[k] = p

        # Update
        S = H @ p @ H.T + R
        K = p @ H.T / S
        m = m + K @ (y[k] - H @ m)
        p = p - K @ S @ K.T

        # Save
        mfs[k] = m
        pfs[k] = p

    # Smoothing pass
    mss = mfs.copy()
    pss = pfs.copy()
    for k in range(num_y - 2, -1, -1):
        (c, low) = scipy.linalg.cho_factor(pps[k + 1])
        G = pfs[k] @ scipy.linalg.cho_solve((c, low), F).T
        mss[k] = mfs[k] + G @ (mss[k + 1] - mps[k + 1])
        pss[k] = pfs[k] + G @ (pss[k + 1] - pps[k + 1]) @ G.T

    return mfs, pfs, mss, pss


if __name__ == "__main__":
    
    np.random.seed(666666)
    
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fouriernc}',
        'font.family': "serif",
        'font.serif': 'New Century Schoolbook',
        'font.size': 18})
    anime_writer = animation.ImageMagickWriter()
    
    # Matern 3/2 coefficients
    ell = 2.
    sigma = 1.
    
    A = np.array([[0., 1.], 
                  [-3 / ell ** 2, -2 * math.sqrt(3) / ell]])
    B = np.array([0., sigma * math.sqrt(12 * math.sqrt(3)) / ell ** (3 / 2)])
    
    m0 = np.zeros((2, ))
    p0 = np.array([[sigma ** 2, 0.], 
                   [0., 3 * sigma ** 2 / ell ** 2]])
    
    dt = 0.1
    F, Q = lti_sde_to_disc(A, B, dt)
    
    H = np.array([[1., 0.]])
    R = 0.1
    
    # Generate data
    T = 100
    ts = np.linspace(dt, T * dt, T)
    xs, ys = simulate_data_from_disc_ss(F, Q, H, R, m0, p0, T)
    
    # Filtering and smoothing
    mfs, pfs, mss, pss = kf_rts(F, Q, H, R, ys, m0, p0)
    
    # Animation for filtering
    # Updating polycollection in matplotlib is difficult,
    # the code in the following for fill_between might look ugly
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    
    line_true, = ax.plot(ts[0], xs[0, 0], c='black', linewidth=3, label='True signal $X(t)$')
    sct_data = ax.scatter(ts[0], ys[0], s=4, c='purple', label='Measurement $Y_k$')
    line_mfs, = ax.plot(ts[0], mfs[0, 0], c='tab:blue', linewidth=3, label='Filtering mean')
    ax.fill_between(
            ts[:1], 
            mfs[:1, 0] - 1.96 * np.sqrt(pfs[:1, 0, 0]), 
            mfs[:1, 0] + 1.96 * np.sqrt(pfs[:1, 0, 0]), 
            color='tab:blue',
            edgecolor='none',
            alpha=0.1,
            label='.95 confidence'
            )
    k_line = ax.axvline(ts[0], c='black', linestyle='--')
    k_text = ax.text(ts[0], 0., '$k=0$', fontsize=18)
    
    ax.set_xlim(0, T * dt)
    ax.set_ylim(-2.5, 1.5)
    ax.legend(loc='upper left', ncol=2, fontsize=18)
    ax.set_xlabel('$t$', fontsize=18)
    
    plt.subplots_adjust(top=.986, bottom=.084, left=.063, right=.988)

    def anime_func(frame):
        line_true.set_data(ts[:frame], xs[:frame, 0])
        line_mfs.set_data(ts[:frame], mfs[:frame, 0])
        k_line.set_data((ts[frame-1], ts[frame-1]), (0, 1))
        k_text.set_text(f'$k={frame}$')
        k_text.set_position((ts[frame], 0.))
        ax.collections.clear()
        ax.fill_between(
            ts[:frame], 
            mfs[:frame, 0] - 1.96 * np.sqrt(pfs[:frame, 0, 0]), 
            mfs[:frame, 0] + 1.96 * np.sqrt(pfs[:frame, 0, 0]), 
            color='tab:blue',
            edgecolor='none',
            alpha=0.1
            )
        ax.scatter(ts[:frame], ys[:frame], s=4, c='purple')
        
    ani = FuncAnimation(fig, anime_func,
                        frames=T, interval=10,
                        repeat=False)

    ani.save('../figs/animes/filter.png', writer=anime_writer)
    # plt.show()
            
    plt.close(fig)
    
    # Animation for smoothing
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    
    ax.plot(ts, xs[:, 0], c='black', linewidth=3, label='True signal $X(t)$')
    ax.scatter(ts, ys, s=4, c='purple', label='Measurement $Y_k$')
    line_mss, = ax.plot(ts, mss[:, 0], c='tab:blue', linewidth=3, label='Smoothing mean')
    ax.fill_between(
            ts[:1], 
            mss[:1, 0] - 1.96 * np.sqrt(pss[:1, 0, 0]), 
            mss[:1, 0] + 1.96 * np.sqrt(pss[:1, 0, 0]), 
            color='tab:blue',
            edgecolor='none',
            alpha=0.1,
            label='.95 confidence'
            )
    k_line = ax.axvline(ts[0], c='black', linestyle='--')
    k_text = ax.text(ts[0], 0., '$k=0$', fontsize=18)
    
    ax.set_xlim(0, T * dt)
    ax.set_ylim(-2.5, 1.5)
    ax.legend(loc='upper left', ncol=2, fontsize=18)
    ax.set_xlabel('$t$', fontsize=18)
    
    plt.subplots_adjust(top=.986, bottom=.084, left=.063, right=.988)

    def anime_func(frame):
        line_mss.set_data(ts[:frame], mss[:frame, 0])
        k_line.set_data((ts[frame-1], ts[frame-1]), (0, 1))
        k_text.set_text(f'$k={frame}$')
        k_text.set_position((ts[frame], 0.))
        ax.collections.clear()
        ax.fill_between(
            ts[:frame], 
            mss[:frame, 0] - 1.96 * np.sqrt(pss[:frame, 0, 0]), 
            mss[:frame, 0] + 1.96 * np.sqrt(pss[:frame, 0, 0]), 
            color='tab:blue',
            edgecolor='none',
            alpha=0.1
            )
        ax.scatter(ts, ys, s=4, c='purple')
        
    ani = FuncAnimation(fig, anime_func,
                        frames=T, interval=10,
                        repeat=False)
    
    ani.save('../figs/animes/smoother.png', writer=anime_writer)
    # plt.show()

