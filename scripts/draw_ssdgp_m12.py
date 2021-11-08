# Draw Matern 1/2 SS-DGP samples and generate Figure 4.4 in the thesis
#
# Zheng Zhao
#
import os
import math
import numpy as np
import matplotlib.pyplot as plt

l2 = 2.
s2 = 2.
l3 = 2.
s3 = 2.


def g(x):
    """Transformation function
    """
    return np.exp(x)


def a_b(x):
    """Return SDE drift and dispersion function a and b
    """
    return -1 / np.array([g(x[1]), l2, l3]) * x, \
           math.sqrt(2) * np.diag([g(x[2]) / np.sqrt(g(x[1])), s2 / np.sqrt(l2), s3 / np.sqrt(l3)])


def euler_maruyama(x0, dt, num_steps, int_steps):
    xx = np.zeros(shape=(num_steps, x0.shape[0]))
    x = x0
    ddt = dt / int_steps
    for i in range(num_steps):
        for j in range(int_steps):
            ax, bx = a_b(x)
            x = x + ax * ddt + np.sqrt(ddt) * bx @ np.random.randn(3)
        xx[i] = x
    return xx


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
    num_steps = 1000
    int_steps = 10
    dt = end_T / num_steps
    tt = np.linspace(dt, end_T, num_steps)

    num_mc = 2

    # Compute Euler--Maruyama
    xx = np.zeros(shape=(num_mc, num_steps, 3))
    for mc in range(num_mc):
        x0 = np.random.randn(3)
        xx[mc] = euler_maruyama(x0, dt, num_steps=num_steps, int_steps=int_steps)

    colours = ('black', 'tab:blue', 'tab:purple')
    markers = ('.', 'x', '1')

    # Plot u
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(12, 13), sharex=True)
    for mc in range(num_mc):
        ax1.plot(tt, xx[mc, :, 0],
                 linewidth=2, c=colours[mc],
                 marker=markers[mc], markevery=200, markersize=16,
                 label=f'Sample {mc + 1}')

    ax1.grid(linestyle='--', alpha=0.3, which='both')

    ax1.set_ylabel('$U^1_0(t)$')
    ax1.set_xlim(0, end_T)
    ax1.set_xticks(np.arange(0, end_T + 1, 1))

    ax1.legend(ncol=2, loc='upper left', fontsize=18)

    # Plot ell
    for mc in range(num_mc):
        ax2.plot(tt, xx[mc, :, 1],
                 linewidth=2, c=colours[mc],
                 marker=markers[mc], markevery=200, markersize=16,
                 label=f'Sample {mc}')

    ax2.grid(linestyle='--', alpha=0.3, which='both')

    ax2.set_ylabel('$U^2_1(t)$')
    ax2.set_xlim(0, end_T)
    ax2.set_xticks(np.arange(0, end_T + 1, 1))

    # Plot sigma
    for mc in range(num_mc):
        ax3.plot(tt, xx[mc, :, 2],
                 linewidth=2, c=colours[mc],
                 marker=markers[mc], markevery=200, markersize=16,
                 label=f'Sample {mc}')

    ax3.grid(linestyle='--', alpha=0.3, which='both')

    ax3.set_xlabel('$t$')
    ax3.set_ylabel('$U^3_1(t)$')
    ax3.set_xlim(0, end_T)
    ax3.set_xticks(np.arange(0, end_T + 1, 1))

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(bottom=0.053)
    plt.savefig(os.path.join(path_figs, 'samples_ssdgp_m12.pdf'))
