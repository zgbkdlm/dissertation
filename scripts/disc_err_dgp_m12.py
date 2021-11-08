# Compare different discretisation schemes on an SS-DGP and Generate Figure 4.3 in the thesis
# This is done purely in sympy and numpy.
#
# Zheng Zhao 2020
#
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

import tme.base_sympy as tme
from sympy import lambdify

l2 = 1.
s2 = 0.1
l3 = 1.
s3 = 0.1


# Transformation function
def g(x):
    return np.exp(0.5 * x)


def g_sym(x):
    return sp.exp(0.5 * x)


# SS-DGP drift
def a(x):
    return -np.array([x[0] / g(x[1]),
                      x[1] / l2,
                      x[2] / l3])


# SS-DGP dispersion
def b(x):
    return math.sqrt(2) * np.diag([g(x[2]) / np.sqrt(g(x[1])),
                                   s2 / math.sqrt(l2),
                                   s3 / math.sqrt(l3)])


def euler_maruyama(x0, dt, dws):
    xx = np.zeros(shape=(dws.shape[0], x0.shape[0]))
    x = x0
    for idx, dw in enumerate(dws):
        x = x + a(x) * dt + b(x) @ dw
        xx[idx] = x
    return xx


# Locally conditional discretisation method for giving
# x_k \approx F(x_{k-1}) + q(x_{k-1})
def lcd_F(x, dt):
    return np.diag(np.exp(-dt / np.array([g(x[1]), l2, l3])))


def lcd_Q(x, dt):
    return np.diag([g(x[2]) ** 2 * (1 - np.exp(-2 * dt / g(x[1]))),
                    s2 ** 2 * (1 - np.exp(-2 * dt / l2)),
                    s3 ** 2 * (1 - np.exp(-2 * dt / l3))])


def lcd(x0, dt, dws):
    """Locally conditional discretisation for Matern 1/2 SS-DGPs
    """
    xx = np.zeros(shape=(dws.shape[0], x0.shape[0]))
    x = x0
    for idx, dw in enumerate(dws):
        x = lcd_F(x, dt) @ x + np.sqrt(lcd_Q(x, dt)) @ dw / np.sqrt(dt)
        xx[idx] = x
    return xx


def local_sum(x, factor):
    target_shape = (int(x.shape[0] / factor), x.shape[1])
    xx = np.zeros(target_shape)
    for i in range(target_shape[0]):
        xx[i] = np.sum(x[i * factor:(i + 1) * factor], axis=0)
    return xx


def give_tme_symbols(order=3, simp=True):
    """Give mean and covariance symbols of Taylor moment expansion.
    """
    # Symbols
    x = sp.MatrixSymbol('x', 3, 1)
    a_sym = -sp.Matrix([x[0] / g_sym(x[1]),
                        x[1] / sp.S(l2),
                        x[2] / sp.S(l3)])
    Q_sym = sp.eye(3)
    b_sym = sp.sqrt(sp.S(2)) * sp.Matrix([[g_sym(x[2]) / sp.sqrt(g_sym(x[1])), 0, 0],
                                          [0, sp.S(s2) / sp.sqrt(sp.S(l2)), 0],
                                          [0, 0, sp.S(s3) / sp.sqrt(sp.S(l3))]
                                          ])
    dt_sym = sp.Symbol('dt')
    tme_mean, tme_cov = tme.mean_and_cov(x, a_sym, b_sym, Q_sym, dt_sym,
                                         order=order, simp=simp)
    tme_mean_func = lambdify([x, dt_sym], tme_mean, 'numpy')
    tme_cov_func = lambdify([x, dt_sym], tme_cov, 'numpy')
    return tme_mean_func, tme_cov_func


def tme_disc(x0, dt, dws, f, Q):
    """Taylor moment expansion discretisation. This gives you a demonstration how to use TME in sympy, however,
    please use the Jax implementation in practice.
    """
    xx = np.zeros(shape=(dws.shape[0], x0.shape[0], 1))
    x = x0.reshape(-1, 1)
    for idx, dw in enumerate(dws):
        x = f(x, dt) + (np.linalg.cholesky(Q(x, dt)) @ dw / np.sqrt(dt))[:, None]
        xx[idx] = x
    return xx[:, :, 0]


def abs_err(x1, x2):
    return np.sum(np.abs(x1 - x2))


if __name__ == '__main__':
    path_figs = '../thesis/figs'
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fouriernc}',
        'font.family': "serif",
        'font.serif': 'New Century Schoolbook',
        'font.size': 20})

    np.random.seed(2020)

    end_T = 10
    num_steps = 100
    dt = end_T / num_steps
    tt = np.linspace(dt, end_T, num_steps)

    boost_factor = 1000
    boost_num_steps = num_steps * boost_factor
    boost_dt = dt / boost_factor
    boost_tt = np.linspace(boost_dt, end_T, boost_num_steps)
    boost_dws = np.sqrt(dt) * np.random.randn(boost_num_steps, 3)

    x0 = np.zeros(shape=(3,))

    # Compute very accurate discretisation
    boost_xx = euler_maruyama(x0, boost_dt, boost_dws)
    exact_xx = boost_xx[boost_factor - 1::boost_factor]
    exact_tt = boost_tt[boost_factor - 1::boost_factor]

    # Compute Euler Maruyama
    dws = local_sum(boost_dws, boost_factor)
    em_xx = euler_maruyama(x0, dt, dws)

    # Compute locally conditional discretisation
    lcd_xx = lcd(x0, dt, dws)

    # Compute TME
    tme_order = 3
    tme_mean_func, tme_cov_func = give_tme_symbols(order=tme_order, simp=True)
    tme_xx = tme_disc(x0, dt, dws, tme_mean_func, tme_cov_func)

    # Compute abs error
    err_dim = 0
    err_em = abs_err(em_xx[:, err_dim], exact_xx[:, err_dim])
    err_lcd = abs_err(lcd_xx[:, err_dim], exact_xx[:, err_dim])
    err_tme = abs_err(tme_xx[:, err_dim], exact_xx[:, err_dim])
    print(f'Euler--Maruyama abs err: {err_em}')
    print(f'LCD abs err: {err_lcd}')
    print(f'TME abs err: {err_tme}')

    # Plot
    plt.figure(figsize=(16, 8))
    plt.plot(tt, exact_xx[:, 0],
             c='black', linewidth=3, label='Numerical exact')
    plt.plot(tt, em_xx[:, 0],
             c='tab:blue', linewidth=3, linestyle=(0, (1, 1)),
             label=f'Euler--Maruyama (abs. err. $\\approx$ {err_em:.1f})')
    plt.plot(tt, lcd_xx[:, 0],
             c='tab:purple', linewidth=3, linestyle=(0, (5, 1)),
             label=f'LCD (abs. err. $\\approx$ {err_lcd:.1f})')
    plt.plot(tt, tme_xx[:, 0],
             c='tab:red', linewidth=3, linestyle=(0, (3, 1, 1, 1)),
             label=f'TME-{tme_order} (abs. err. $\\approx$ {err_tme:.1f})')

    plt.legend(loc='lower left', fontsize=24)

    plt.grid(linestyle='--', alpha=0.3, which='both')
    plt.xlim(0, end_T)

    plt.xlabel('$t$', fontsize=26)
    plt.ylabel('$U^1_0(t)$', fontsize=26)
    plt.xticks(np.arange(0, end_T + 1, 1))

    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(path_figs, 'disc-err_dgp_m12.pdf'))
