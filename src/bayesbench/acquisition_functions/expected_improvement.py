"""
This module implements optimization using Gaussian Process with 
Expected Improvement acquisition function.
https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html 
EI(x) = E[max(f_best - f(x), 0)] acquisition function for Bayesian optimization.
"""
import numpy as np
from scipy.stats import norm

def expected_improvement(mean: np.ndarray, std: np.ndarray, 
                         f_best: float, xi: float = 0.01 )-> np.ndarray:
        """
        computes the expected improvement acquisition function for Bayesian optimization.
        returns the expected improvement for each candidate point.
        mean : mean of the GP at the candidate points
        std : standard deviation of the GP at the candidate points
        f_best : best observed value so far
        xi : exploration-exploitation trade-off parameter
        """
        improvement = f_best - mean - xi
        ei = np.zeros_like(mean)
        nonzero_std = std > 0
        z = np.zeros_like(mean)
        z[nonzero_std] = improvement[nonzero_std] / std[nonzero_std]
        
        ei[nonzero_std] = improvement[nonzero_std] * norm.cdf(z[nonzero_std]) 
        + std[nonzero_std] * norm.pdf(z[nonzero_std])
        return ei