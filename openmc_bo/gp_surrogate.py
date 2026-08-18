'''
 gp_surrogate.py - Gaussian Process surrogate model
---------------------------------

Matern-5/2 kernel with ARD length scales, WhiteKernel noise and StandardScaler pre-processing.

'''

from __future__ import annotations
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from typing import Tuple

from .config import RunConfig


def train_gp(
    X: np.ndarray,
    y: np.ndarray,
    cfg: RunConfig,
) -> Tuple[GaussianProcessRegressor, StandardScaler]:
    '''
    Fit a Matern-5/2 GP with ARD length scales.

    '''
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale        = np.ones(cfg.n_vars),
            length_scale_bounds=(1e-3, 10),           
            nu = 2.5,
          )
        + WhiteKernel(
            noise_level        = 1e-5,
            noise_level_bounds = (1e-8, 1e-2),
          )
    )

    gp = GaussianProcessRegressor(
        kernel               = kernel,
        n_restarts_optimizer = cfg.gp_restarts,
        normalize_y          = True,
        alpha                = 1e-8,
    )
    gp.fit(Xs, y)
    return gp, scaler


def predict(
    X_pred: np.ndarray,
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    return_std: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    GP prediction on new points.

    '''
    Xs = scaler.transform(X_pred)
    if return_std:
        mu, sigma = gp.predict(Xs, return_std=True)
    else:
        mu    = gp.predict(Xs, return_std=False)
        sigma = np.zeros_like(mu)
    return mu, sigma


def expected_improvement(
    X_cand: np.ndarray,
    gp: GaussianProcessRegressor,
    scaler: StandardScaler,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    '''
    Expected Improvement acquisition function (maximisation).

    
    Here, ei = (n_candidates,) ––> higher means more promising candidate
    
    '''
    mu, sigma = predict(X_cand, gp, scaler)
    sigma     = np.maximum(sigma, 1e-9)
    imp = mu - y_best - xi
    Z   = imp / sigma
    ei  = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei


def gp_summary(gp: GaussianProcessRegressor, cfg: RunConfig) -> str:
    
    '''Return a human-readable summary of the fitted GP kernel.'''
    
    lines = [f"GP surrogate — {len(gp.X_train_)} training points"]
    lines.append(f"  Kernel : {gp.kernel_}")
    lines.append(f"  Log-marginal-likelihood : {gp.log_marginal_likelihood_value_:.4f}")
    return "\n".join(lines)
