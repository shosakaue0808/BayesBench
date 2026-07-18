"""
This file has functions to randomly sample points from domain of objective functions, and 
get samples as (x, y) pairs.
This is baseline compared with GP-BO
"""
import numpy as np
import random
from collections.abc import Callable

def sample_uniform(bounds: np.ndarray, rng: np.random.Generator)->np.ndarray:
    """
    sample one (x1, x2, ... xd) point uniformly from bounds
    """
    return rng.uniform(bounds[:, 0], bounds[:, 1])

def random_search(objective: Callable[[np.ndarray], float], 
                  bounds: np.ndarray, budget: int, rng: np.random.Generator)-> tuple[np.ndarray, np.ndarray]:
    """
    Run random search on a blackbox objective. 
    Parameters
    ----------
    objective:
        Function to minimize.
    bounds:
        Array with shape (d, 2), where each row is [lower, upper].
    budget:
        Number of objective evaluations.
    ----------------------------------------------------------------
    Returns
    tuple[np.ndarray, np.ndarray]
        X: sampled input points
        Y: value of function at input points   
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    
    X = []
    y = []

    for _ in range(budget):
        x = sample_uniform(bounds, rng)
        value = objective(x)

        X.append(x)
        y.append(value)

    return np.array(X), np.array(y)