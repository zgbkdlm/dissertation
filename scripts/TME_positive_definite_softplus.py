# This will generate Figure 3.1 in the thesis.
#
# Zheng Zhao 2020
#
import os
import sympy
import numpy as np
import matplotlib.pyplot as plt
import tme.base_sympy as tme

from sympy import lambdify
from matplotlib.ticker import MultipleLocator

if __name__ == '__main__':

    # Initial value and paras
    np.random.seed(666)
    x0 = np.array([[0.],
                   [0.]])

    # Example SDE
    kappa = sympy.Symbol('k')
    x = sympy.MatrixSymbol('x', 2, 1)
    f = sympy.Matrix([[sympy.log(1 + sympy.exp(x[0])) + kappa * x[1]],
                      [sympy.log(1 + sympy.exp(x[1])) + kappa * x[0]]])
    L = sympy.eye(2)
    Q = sympy.eye(2)
    dt_sym = sympy.Symbol('dt')

    # TME
    tme_mean, tme_cov = tme.mean_and_cov(x, f, L, Q, dt_sym,
                                         order=2, simp=True)

    # Cov
    cov_func = lambdify([x, kappa, dt_sym], tme_cov, 'numpy')

    # Compute
    # xx = np.linspace(0, 1, 200)
    kk = np.linspace(-4, 4, 500)
    dts = np.linspace(0.01, 6, 500)

    mineigs = np.zeros((kk.size, dts.size))

    for i in range(kk.size):
        for j in range(dts.size):
            eig, _ = np.linalg.eigh(cov_func(x0, kk[i], dts[j]))
            mineigs[i, j] = eig.min()

    # Plot
    path_figs = '../thesis/figs'
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fouriernc}',
        'font.family': "serif",
        'font.serif': 'New Century Schoolbook',
        'font.size': 18})

    fig = plt.figure()

    grid_k, grid_dt = np.meshgrid(dts, kk)

    mineigs_crop = mineigs
    mineigs_crop[mineigs_crop < 0] = -1.

    ax = plt.axes()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    cnt = plt.contourf(grid_dt, grid_k, mineigs_crop, cmap=plt.cm.Blues_r)
    for c in cnt.collections:
        c.set_edgecolor("face")

    cbar = plt.colorbar()
    cbar_ticks = [tick.get_text() for tick in cbar.ax.get_yticklabels()]
    cbar_ticks[0] = '<0'

    cbar.ax.set_yticklabels(cbar_ticks)

    plt.axvline(-0.5, c='red', linestyle='--', alpha=0.5)
    plt.axvline(0.5, c='red', linestyle='--', alpha=0.5)

    plt.xlabel(r'$\kappa$')
    plt.ylabel(r'$\Delta t$')
    plt.title(r'$\lambda_{\mathrm{min}}(\Sigma_2(\Delta t))$')

    plt.tight_layout(pad=0.1)

    # plt.show()

    plt.savefig(os.path.join(path_figs, 'tme-softplus-mineigs.pdf'))
