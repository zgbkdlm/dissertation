# Show the GP regression on a magnitude-varying rectangular wave signal, and generate Figure 1.1.
#
# Zheng Zhao 2021
#
import os
import math
import numpy as np
import gpflow
import matplotlib.pyplot as plt

from typing import Tuple
from matplotlib.ticker import MultipleLocator

path_figs = '../thesis/figs'
np.random.seed(666)
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{fouriernc}',
    'font.family': "serif",
    'font.serif': 'New Century Schoolbook',
    'font.size': 20})


def rect(t: np.ndarray,
         r: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rectangle signal. Return the signal and an noisy measurement of it.
    """
    tau = (t - np.min(t)) / (np.max(t) - np.min(t))

    p = np.linspace(1 / 6, 5 / 6, 5)

    y = np.zeros_like(t)
    y[(tau >= 0) & (tau < p[0])] = 0
    y[(tau >= p[0]) & (tau < p[1])] = 1
    y[(tau >= p[1]) & (tau < p[2])] = 0
    y[(tau >= p[2]) & (tau < p[3])] = 0.6
    y[(tau >= p[3]) & (tau < p[4])] = 0
    y[tau >= p[4]] = 0.4

    return y, y + math.sqrt(r) * np.random.randn(*t.shape)


# Simulate measurements
t = np.linspace(0, 1, 400).reshape(-1, 1)
r = 0.004
ft, y = rect(t, r)

# GPflow
ell = 1.
sigma = 1.

m12 = gpflow.kernels.Matern12(lengthscales=ell, variance=sigma)
m32 = gpflow.kernels.Matern32(lengthscales=ell, variance=sigma)
m52 = gpflow.kernels.Matern52(lengthscales=ell, variance=sigma)
rbf = gpflow.kernels.SquaredExponential(lengthscales=ell, variance=sigma)

# Plots
for name, label, cov in zip(['m12', 'm32', 'm52', 'rbf'],
                            [r'Mat\'ern $1\,/\,2$', r'Mat\'ern $3\,/\,2$', r'Mat\'ern $5\,/\,2$', r'RBF'],
                            [m12, m32, m52, rbf]):
    print(f'GP regression with {name} cov function')
    model = gpflow.models.GPR(data=(t, y), kernel=cov, mean_function=None)
    model.likelihood.variance.assign(r)

    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(model.training_loss, model.trainable_variables,
                            method='L-BFGS-B',
                            options={'disp': True})

    m, P = model.predict_f(t)

    # Plot and save
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes()
    plt.plot(t, ft, c='black', alpha=0.8, linestyle='--', linewidth=2, label='True signal')
    plt.scatter(t, y, s=15, c='black', edgecolors='none', alpha=0.3, label='Measurements')
    plt.plot(t, m, c='black', linewidth=3, label=label)
    plt.fill_between(
        t[:, 0],
        m[:, 0] - 1.96 * np.sqrt(P[:, 0]),
        m[:, 0] + 1.96 * np.sqrt(P[:, 0]),
        color='black',
        edgecolor='none',
        alpha=0.2,
    )

    plt.grid(linestyle='--', alpha=0.3, which='both')

    plt.xlim(0, 1)
    plt.ylim(-0.2, 1.2)

    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter('{x:.1f}')

    ax.yaxis.set_major_locator(MultipleLocator(0.4))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter('{x:.1f}')

    plt.xlabel('$t$', fontsize=24)
    plt.title('$\\ell \\approx {:.2f}, \\quad \\sigma \\approx {:.2f}$'.format(cov.lengthscales.numpy(),
                                                                               cov.variance.numpy()))
    plt.legend(loc='upper right', fontsize='large')

    plt.tight_layout(pad=0.1)

    filename = 'gp-fail-example-' + name + '.pdf'
    plt.savefig(os.path.join(path_figs, filename))
