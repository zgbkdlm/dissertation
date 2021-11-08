# Demonstrate TME on a Benes model for some expectation approximations. This will generate Figure 3.1 in the thesis.
#
# Zheng Zhao 2020
#
import os
import math
import sympy
import numpy as np
import matplotlib.pyplot as plt
import tme.base_sympy as tme

from typing import Callable
from sympy import lambdify


def rieman1D(x: np.ndarray,
             f: Callable,
             *args, **kwargs):
    r"""Riemannian computation of an integral
    \int f(x, *args, **kwargs) dx \approx \sum f(x_i) (x_i - x_i-1).
    Can be replaced by :code:`np.trapz`.
    """
    return np.sum(f(x[1:], *args, **kwargs) * np.diff(x))


def benesPDF(x: np.ndarray,
             x0: float,
             dt: float):
    """
    Transition density of the Benes model.
    See, pp. 214 of Sarkka 2019.
    """
    return 1 / math.sqrt(2 * math.pi * dt) * np.cosh(x) / np.cosh(x0) \
           * math.exp(-0.5 * dt) * np.exp(-0.5 / dt * (x - x0) ** 2)


def f_mean(x: np.ndarray,
           x0: float,
           dt: float):
    """Expectation integrand.
    """
    return x * benesPDF(x, x0, dt)


def f_x2(x: np.ndarray,
         x0: float,
         dt: float):
    """Expectation integrand.
    """
    return x ** 2 * benesPDF(x, x0, dt)


def f_x3(x: np.ndarray,
         x0: float,
         dt: float):
    """Expectation integrand.
    """
    return x ** 3 * benesPDF(x, x0, dt)


def f_nonlinear(x: np.ndarray,
                x0: float,
                dt: float):
    """Expectation integrand.
    """
    return np.sin(x) * benesPDF(x, x0, dt)


def softplus(x):
    return np.log(1 + np.exp(x))


def softplus_sympy(x):
    return sympy.log(1 + sympy.exp(x))


def f_nn(x: np.ndarray,
         x0: float,
         dt: float):
    """
    A toy-level neural network with a single perceptron
    NN(x) = sigmoid(x)
    """
    return softplus(softplus(x)) * benesPDF(x, x0, dt)


def em_mean(f: Callable,
            x0: float,
            dt: float):
    """E[x | x0] by Euler Maruyama
    """
    return x0 + f(x0) * dt


def em_cov(f: Callable,
           x0: float,
           dt: float):
    """Var[x | x0] by Euler Maruyama
    """
    return dt


def em_x3(f: Callable,
          x0: float,
          dt: float):
    """E[x^3 | x0] by Euler Maruyama
    """
    return x0 ** 3 + 3 * x0 ** 2 * f(x0) * dt \
           + 3 * x0 * dt + f(x0) ** 3 * dt ** 3 \
           + 3 * f(x0) * dt ** 2


def ito15_mean(f: Callable,
               dfdx: Callable,
               d2fdx2: Callable,
               x0: float,
               dt: float):
    """E[x | x0] by Ito-1.5
    """
    return x0 + f(x0) * dt + (dfdx(x0) * f(x0) + 0.5 * d2fdx2(x0)) * dt ** 2 / 2


def ito15_cov(f: Callable,
              dfdx: Callable,
              d2fdx2: Callable,
              x0: float,
              dt: float):
    """Cov[x | x0] by Ito-1.5
    """
    return dt + dfdx(x0) ** 2 * dt ** 3 / 3 + dfdx(x0) * dt ** 2


def ito15_x3(f: Callable,
             dfdx: Callable,
             d2fdx2: Callable,
             x0: float,
             dt: float):
    """E[x^3 | x0] by Ito-1.5
    """
    z = x0 + f(x0) * dt + (dfdx(x0) * f(x0) + 0.5 * d2fdx2(x0)) * dt ** 2 / 2
    return z ** 3 + 3 * z * dt + 3 * z * dfdx(x0) * dt ** 2 + z * dfdx(x0) * dt ** 3


tanh = lambda u: np.tanh(u)
dtanh = lambda u: 1 - np.tanh(u) ** 2
ddtanh = lambda u: 2 * (np.tanh(u) ** 3 - np.tanh(u))

if __name__ == '__main__':

    # Initial value and paras
    np.random.seed(666)
    x0 = 0.5
    T = np.linspace(0.01, 4, 200)

    # Riemannian range
    range_dx = np.linspace(x0 - 20, x0 + 20, 100000)

    # Benes SDE
    x = sympy.MatrixSymbol('x', 1, 1)
    f = sympy.Matrix([sympy.tanh(x[0])])
    L = sympy.eye(1)
    Q = sympy.eye(1)
    dt_sym = sympy.Symbol('dt')

    # TME
    tme_mean, tme_cov = tme.mean_and_cov(x, f, L, Q, dt_sym,
                                         order=3, simp=True)
    tme_x3 = tme.expectation(sympy.Matrix([x[0] ** 3]), x, f, L, Q, dt_sym,
                             order=3, simp=True)
    tme_nonlinear3 = tme.expectation(sympy.Matrix([sympy.sin(x[0])]), x, f, L, Q, dt_sym,
                                     order=2, simp=True)
    tme_nonlinear4 = tme.expectation(sympy.Matrix([sympy.sin(x[0])]), x, f, L, Q, dt_sym,
                                     order=3, simp=True)
    tme_nn = tme.expectation(sympy.Matrix([softplus_sympy(softplus_sympy(x[0]))]), x, f, L, Q, dt_sym,

                             order=3, simp=True)

    tme_mean_func = lambdify([x, dt_sym], tme_mean, 'numpy')
    tme_cov_func = lambdify([x, dt_sym], tme_cov, 'numpy')
    tme_x3_func = lambdify([x, dt_sym], tme_x3, 'numpy')
    tme_nonlinear_func3 = lambdify([x, dt_sym], tme_nonlinear3, 'numpy')
    tme_nonlinear_func4 = lambdify([x, dt_sym], tme_nonlinear4, 'numpy')
    tme_nonlinear_nn = lambdify([x, dt_sym], tme_nn, 'numpy')

    # Result containers
    tme_mean_result = np.zeros_like(T)
    tme_cov_result = np.zeros_like(T)
    tme_x3_result = np.zeros_like(T)
    tme_nonlinear3_result = np.zeros_like(T)
    tme_nonlinear4_result = np.zeros_like(T)
    tme_nn_result = np.zeros_like(T)

    riem_mean_result = np.zeros_like(T)
    riem_cov_result = np.zeros_like(T)
    riem_x3_result = np.zeros_like(T)
    riem_nonlinear_result = np.zeros_like(T)
    riem_nn_result = np.zeros_like(T)

    em_mean_result = np.zeros_like(T)
    em_cov_result = np.zeros_like(T)
    em_x3_result = np.zeros_like(T)

    ito15_mean_result = np.zeros_like(T)
    ito15_cov_result = np.zeros_like(T)
    ito15_x3_result = np.zeros_like(T)

    for idx, t in enumerate(T):
        tme_mean_result[idx] = tme_mean_func(np.array([[x0]]), np.array([[t]]))
        tme_cov_result[idx] = tme_cov_func(np.array([[x0]]), np.array([[t]]))
        tme_x3_result[idx] = tme_x3_func(np.array([[x0]]), np.array([[t]]))
        tme_nonlinear3_result[idx] = tme_nonlinear_func3(np.array([[x0]]), np.array([[t]]))
        tme_nonlinear4_result[idx] = tme_nonlinear_func4(np.array([[x0]]), np.array([[t]]))
        tme_nn_result[idx] = tme_nonlinear_nn(np.array([[x0]]), np.array([[t]]))

        riem_mean_result[idx] = rieman1D(range_dx, f_mean, x0=x0, dt=t)
        riem_cov_result[idx] = rieman1D(range_dx, f_x2, x0=x0, dt=t) - riem_mean_result[idx] ** 2
        riem_x3_result[idx] = rieman1D(range_dx, f_x3, x0=x0, dt=t)
        riem_nonlinear_result[idx] = rieman1D(range_dx, f_nonlinear, x0=x0, dt=t)
        riem_nn_result[idx] = rieman1D(range_dx, f_nn, x0=x0, dt=t)

        em_mean_result[idx] = em_mean(lambda z: np.tanh(z), x0, t)
        em_cov_result[idx] = em_cov(lambda z: np.tanh(z), x0, t)
        em_x3_result[idx] = em_x3(lambda z: np.tanh(z), x0, t)

        ito15_mean_result[idx] = ito15_mean(tanh, dtanh, ddtanh, x0, t)
        ito15_cov_result[idx] = ito15_cov(tanh, dtanh, ddtanh, x0, t)
        ito15_x3_result[idx] = ito15_x3(tanh, dtanh, ddtanh, x0, t)

    # Plot
    path_figs = '../thesis/figs'
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fouriernc}',
        'font.family': "serif",
        'font.serif': 'New Century Schoolbook',
        'font.size': 15})

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(9, 12), sharex=True)

    # No need to show the mean because the results are identical
    # plt.figure()
    # plt.plot(T, tme_mean_result, label='TME')
    # plt.plot(T, riem_mean_result, label='Exact')
    # plt.plot(T, em_mean_result, label='EM')
    # plt.plot(T, ito15_mean_result, label='Ito15')
    # plt.legend()
    # plt.savefig(os.path.join(path_figs, 'tme-benes-mean.pdf'))

    # Variance
    axs[0].plot(T, riem_cov_result,
                c='black',
                linewidth=3, marker='+', markevery=20,
                markersize=17,
                label='Exact')
    axs[0].plot(T, tme_cov_result,
                c='tab:blue',
                linewidth=3, marker='.', markevery=20,
                markersize=17,
                label='TME-3')
    axs[0].plot(T, em_cov_result,
                c='tab:red',
                linewidth=3, marker='1', markevery=20,
                markersize=17,
                label='Euler--Maruyama')
    axs[0].plot(T, ito15_cov_result,
                c='tab:purple',
                linewidth=3, marker='2', markevery=20,
                markersize=17,
                label=r'It\^{o}-1.5')

    axs[0].grid(linestyle='--', alpha=0.3, which='both')

    axs[0].set_ylabel(r'$\mathrm{Var}\,[X(t) \mid X(t_0)]$')
    axs[0].legend(loc='upper left', fontsize=17)

    # X^3
    axs[1].plot(T, riem_x3_result,
                c='black',
                linewidth=3, marker='+', markevery=20,
                markersize=17,
                label='Exact')
    axs[1].plot(T, tme_x3_result,
                c='tab:blue',
                linewidth=3, marker='.', markevery=20,
                markersize=17,
                label='TME-3')
    axs[1].plot(T, em_x3_result,
                c='tab:red',
                linewidth=3, marker='1', markevery=20,
                markersize=17,
                label='Euler--Maruyama')
    axs[1].plot(T, ito15_x3_result,
                c='tab:purple',
                linewidth=3, marker='2', markevery=20,
                markersize=17,
                label=r'It\^{o}-1.5')

    axs[1].grid(linestyle='--', alpha=0.3, which='both')

    axs[1].set_ylabel(r'$\mathbb{E} \, [X^3(t) \mid X(t_0)]$')
    axs[1].legend(loc='upper left', fontsize=17)

    # Nonlinear function only by TME
    axs[2].plot(T, riem_nonlinear_result,
                c='black',
                linewidth=3, marker='+', markevery=20,
                markersize=17,
                label='Exact')
    axs[2].plot(T, tme_nonlinear3_result,
                c='tab:purple',
                linewidth=3, marker='.', markevery=20,
                markersize=17,
                label='TME-2')
    axs[2].plot(T, tme_nonlinear4_result,
                c='tab:blue',
                linewidth=3, marker='x', markevery=20,
                markersize=17,
                label='TME-3')

    axs[2].grid(linestyle='--', alpha=0.3, which='both')
    axs[2].set_ylim(-6, 4)

    axs[2].set_ylabel(r'$\mathbb{E}\, [\sin(X(t)) \mid X(t_0)]$')
    axs[2].legend(loc='lower left', fontsize=17)

    # Neural network
    axs[3].plot(T, riem_nn_result,
                c='black',
                linewidth=3, marker='+', markevery=20,
                markersize=17,
                label='Exact')
    axs[3].plot(T, tme_nn_result,
                c='tab:blue',
                linewidth=3, marker='.', markevery=20,
                markersize=17,
                label='TME-3')

    axs[3].grid(linestyle='--', alpha=0.3, which='both')
    # plt.ylim(-1, 1)

    axs[3].set_xlim(0, 4)

    axs[3].set_xlabel('$t$', fontsize=16)
    axs[3].set_ylabel(r'$\mathbb{E}\, [\log(1+\exp(\log(1+\exp(X(t))))) \mid X(t_0)]$')
    axs[3].legend(loc='upper left', fontsize=17)

    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(path_figs, 'tme-benes-all.pdf'))
    # plt.show()
