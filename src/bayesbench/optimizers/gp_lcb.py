import numpy as np
from collections.abc import Callable

from bayesbench.optimizers.random_search import sample_uniform
from bayesbench.optimizers.gp import generate_candidate_points, make_gp_model
from bayesbench.acquisition_functions.lower_confidence_bound import lower_confidence_bound

def gp_lcb(
    objective: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    budget: int,
    rng: np.random.Generator,
    beta: float = 2.0,
    random_state: int= 42,
    n_initial: int = 5,
    n_candidates: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Gaussian Process Bayesian Optimization with Expected Improvement.

    Parameters
    ----------
    objective:
        Objective function to minimize.
    bounds:
        Array with shape (d, 2), where each row is [lower, upper].
    budget:
        Total number of objective evaluations.
    n_initial:
        Number of initial random evaluations before fitting the GP.
    n_candidates:
        Number of candidate points used to approximately maximize EI.
    seed:
        Random seed for reproducibility.
    xi:
        Exploration parameter for Expected Improvement.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X: evaluated input points with shape (budget, d)
        y: objective values with shape (budget,)
    """
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if n_initial <= 0:
        raise ValueError("n_initial must be positive.")
    if n_initial > budget:
        raise ValueError("n_initial must be less than or equal to budget.")
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive.")

    X = []
    y = []


    # 1. Initial random evaluations to generate GP model
    for _ in range(n_initial):
        x = sample_uniform(bounds, rng)
        value = objective(x)

        X.append(x)
        y.append(value)

    # 2. Bayesian optimization loop
    while len(y) < budget:
        X_train = np.array(X)
        y_train = np.array(y)

        # Fit GP surrogate to observed data. This updates the GP model with new observations.
        # the GP has learned a posterior distribution over possible functions 
        # that are consistent with the observed data.
        gp = make_gp_model(random_state=random_state)
        gp.fit(X_train, y_train)

        # Generate candidate points and compute GP predictions
        candidates = generate_candidate_points(
            bounds=bounds,
            n_candidates=n_candidates,
            rng=rng
        )

        # Compute mean and standard deviation of GP predictions (function values) at candidate points
        # doing inference from that posterior distribution.
        mean, std = gp.predict(candidates, return_std=True)

        # Compute lcb on candidates and take best candidate
        lcb = lower_confidence_bound(mean, std, beta = beta)

        # next points to evaluate the objective function
        next_x = candidates[int(np.argmin(lcb))]
        next_y = objective(next_x)

        X.append(next_x)
        y.append(next_y)

    return np.array(X), np.array(y)