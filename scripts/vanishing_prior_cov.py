# Numerically plot the vanishing covariance problem for Matern SS-DGP. This gives Figure 4.8 in the thesis.
#
# Zheng Zhao 2021
#
import os
import math
import numpy as np
import matplotlib.pyplot as plt

l2 = 0.1
s2 = 0.1
l3 = 0.1
s3 = 0.1


def g(x):
    return np.exp(x)


def a_b(x):
    """Return a(x) and b(x)
    """
    return -1 / np.array([g(x[1]), l2, l3]) * x, \
           math.sqrt(2) * np.diag([g(x[2]) / np.sqrt(g(x[1])), s2 / np.sqrt(l2), s3 / np.sqrt(l3)])


def euler_maruyama(x0, dt, num_steps):
    xx = np.zeros(shape=(num_steps, x0.shape[0]))
    x = x0
    for i in range(num_steps):
        ax, bx = a_b(x)
        x = x + ax * dt + np.sqrt(dt) * bx @ np.random.randn(3)
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

    end_T = 1
    num_steps = 1000
    dt = end_T / num_steps
    tt = np.linspace(dt, end_T, num_steps)

    num_mc = 20000

    m0 = np.array([1., 1., 1.])
    P0 = np.array([[1., 0., 0.5],
                   [0., 1., 0.],
                   [0.5, 0., 1.]])
    P0_chol = np.linalg.cholesky(P0)

    # Compute Euler Maruyama
    xx = np.zeros(shape=(num_mc, num_steps, 3))
    for mc in range(num_mc):
        x0 = m0 + P0_chol @ np.random.randn(3)
        xx[mc] = euler_maruyama(x0, dt, num_steps=num_steps)

    Ex = np.mean(xx, axis=0)[:, 0]
    Ey = np.mean(xx, axis=0)[:, 2]
    Exy = np.mean(xx[:, :, 0] * xx[:, :, 2], axis=0)

    cov = Exy - Ex * Ey

    # Plot
    plt.figure(figsize=(7, 4))
    plt.plot(tt, cov, c='black', linewidth=3, label=r'$\mathrm{Cov}\,\big[U^1_0(t), U^3_1(t)\big]$')

    plt.grid(linestyle='--', alpha=0.3, which='both')

    plt.xlabel('$t$')
    plt.ylabel(r'$\mathrm{Cov}\,\big[U^1_0(t), U^3_1(t)\big]$')
    plt.xlim(0, end_T)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    plt.legend(loc='upper right', fontsize=20)

    plt.tight_layout(pad=0.1)

    plt.savefig(os.path.join(path_figs, 'vanishing-cov.pdf'))
    # plt.show()
