"""
Draw GP samples

"""
import jax
import math
import jax.numpy as jnp
import jax.scipy
import jax.scipy.optimize
import matplotlib.pyplot as plt
from jax.config import config

config.update("jax_enable_x64", True)

plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{fouriernc}',
        'font.family': "serif",
        'font.serif': 'New Century Schoolbook',
        'font.size': 18})

# Random seed
key = jax.random.PRNGKey(6666)

jndarray = jnp.ndarray


def m12_cov(t1: float, t2: float, s: float, ell: float) -> float:
    """Matern 1/2"""
    return s ** 2 * jnp.exp(-jnp.abs(t1 - t2) / ell)


def m32_cov(t1: float, t2: float, s: float, ell: float) -> float:
    """Matern 3/2"""
    z = math.sqrt(3) * jnp.abs(t1 - t2) / ell
    return s ** 2 * (1 + z) * jnp.exp(-z)


vectorised_m12_cov = jax.vmap(jax.vmap(m12_cov, in_axes=[0, None, None, None]), in_axes=[None, 0, None, None])
vectorised_m32_cov = jax.vmap(jax.vmap(m32_cov, in_axes=[0, None, None, None]), in_axes=[None, 0, None, None])


# Times
ts = jnp.linspace(0, 1, 1000)
num_mcs = 10

# Paras
s = 1.
ell = 1.

# Compute mean and covariances
mean = jnp.zeros_like(ts)

for cov_func, cov_name in zip([vectorised_m12_cov, vectorised_m32_cov], 
                              ['m12', 'm32']):
    
    cov = cov_func(ts, ts, s, ell)
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    plt.xlim(0, 1)
    plt.ylabel('$U(t)$', fontsize=20)
    plt.xlabel('$t$', fontsize=20)
    
    for i in range(num_mcs):
        
        # Random key
        key, subkey = jax.random.split(key)

        # Draw!
        gp_sample = jax.random.multivariate_normal(key=subkey, mean=mean, cov=cov)

        # Plot
        ax.plot(ts, gp_sample, linewidth=1)
    
    if cov_func is vectorised_m12_cov:
        plt.subplots_adjust(top=0.995, bottom=0.09, left=0.08, right=0.981, hspace=0.2, wspace=0.2)
    else:
        plt.subplots_adjust(top=0.995, bottom=0.09, left=0.105, right=0.981, hspace=0.2, wspace=0.2)
    
    plt.savefig(f'../figs/gp-sample-{cov_name}.pdf')
    plt.cla()

