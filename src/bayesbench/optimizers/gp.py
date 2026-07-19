import numpy as np
from collections.abc import Callable

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from bayesbench.optimizers.random_search import sample_uniform

def make_gp_model(random_state:int = 42) -> GaussianProcessRegressor:
    """
    Create a Gaussian Process surrogate model for Bayesian optimization.
    """
    kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-10, 1e-2))
    )

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state,
    )

def generate_candidate_points(
    bounds: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator
    ) -> np.ndarray:
    """
    Generate random candidate n points (x1, x2) where Expected Improvement is evaluated. 
    This is used to get next sample point to evaluate the objective fucntion
    """
    return np.array(
        [sample_uniform(bounds, rng) for _ in range(n_candidates)]
    )

