"""
Ackley function benchmark for BayesBench. This is a D-dimensional function.
https://www.sfu.ca/~ssurjano/ackley.html
global minimum is at x = (0, 0, ..., 0) with f(x) = 0
"""
import numpy as np

def ackley(x: np.ndarray) -> float:
    """
    Evaluate Ackley function at D-dimensional input point.
    Given D-dimensional input x, return value of the function at that point.
    domain is x_i in [-32.768, 32.768] for all i
    """
    # convert input to np array 
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"Expected x to be 1D array, got {x.ndim}D array.")

    if np.any(x < -32.768) or np.any(x > 32.768):
        raise ValueError("Input is out of domain")

    d = len(x)
    a = 20
    b = 0.2
    c = 2 * np.pi

    sqr_term = np.sqrt(np.sum(x**2)/d)
    sum_sq_term = -a * np.exp(-b*sqr_term)
    cos_term = -np.exp(np.sum(np.cos(c*x))/d)

    return float(sum_sq_term + cos_term + a + np.exp(1))

def ackley_bounds(d: int) -> np.ndarray:
    """
    return the domain of ackley function for d dimensions
    """
    return np.array([[-32.768, 32.768]]*d, dtype=float)

def ackley_global_minimum() -> float:
    """
    return global minimum of ackley function
    """
    return 0.0