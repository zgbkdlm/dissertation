This folder contains Python/Matlab scripts that generate some of the figures in the dissertation. Specifically, the scripts in this folder are as follows.

1. `disc_err_dgp_m12.py`: Compute the discretisation errors for a Matern DGP. Related to **Figure 4.3**.

2. `draw_ssdgp_m12.py`: Draw samples from a Matern 1/2 SS-DGP. Related to **Figure 4.4**.

3. `draw_ssdgp_m32.py`: Draw samples from a Matern 3/2 SS-DGP. Related to **Figure 4.5**.

4. `showcase_gp_rectangular.py`: Perform GP regression for a rectangular signal. Related to **Figure 1.1**.

5. `showcase_gp_sinusoidal.py`: Perform GP regression for a sinusoidal signal. Related to **Figure 1.1**.

6. `spectro_temporal.py`: Spectro-temporal state-space method for estimation of spectrogram. Related to **Figure 5.2**.

7. `TME_estimation_benes.py`: Use TME to estimate a few expectations of a Benes SDE. Related to **Figure 3.1**.

8. `TME_postive_definite_softplus.py`: Analyse the postive definiteness of the TME covariance estimator for an SDE. Related to **Figure 3.2**.

9. `vanishing_prior_cov.py`: Estimate a cross-covariance of an SS-DGP. Related to **Figure 4.8**.

# Requirements

In order to run the scripts, you need to install a few packages as follows.

`pip install numpy scipy scikit-learn sympy tme matplotlib`.

Additionally, if you want to run scripts `showcase_gp_rectangular.py` and `showcase_gp_sinusoidal.py`, you need to install `gpflow`, that is, `pip install gpflow`.

# License

You are free to do anything you want with the scripts in this folder, except that `spectro_temporal.py` is under the MIT license.
