"""
Branin benchmark function. 2D, non-convex, global minimum are known 
https://www.sfu.ca/~ssurjano/branin.html

This file defines the "true" objective function for the first BayesBench
optimization benchmark. Random search and Bayesian optimization both evaluate
this function, while Bayesian optimization also fits a surrogate model to the
observed input-output pairs.

Goal:
    minimize branin(x)

Standard domain:
    x1 in [-5, 10]
    x2 in [0, 15]

Known global minimum:
    f(x*) ≈ 0.397887
"""

import numpy as np

def branin(x: np.ndarray) -> float:
    """
    Evaluate Branin-Hoo function at 2D input point.
    Given 2D input x, return value of the function at that point.
    domain is x1 in [-5, 10]
              x2 in [0, 15]            
    """
    x = np.asarray(x, dtype=float)

    if x.shape != (2,):
        raise ValueError(f"Expected x to have shape (2,), got {x.shape}.")

    x1, x2 = x

    if(x1 > 10 or x1 <-5):
        raise ValueError("x1 is out of domain")
    if(x2 > 15 or x2 < 0):
        raise ValueError("x2 is out of domain")
    a = 1
    b = 5.1/(4.0 * np.pi**2)
    c= 5/np.pi
    r = 6
    s =10
    t = 1/(8*np.pi)
    return float(a*(x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * np.cos(x1) + s)

def branin_bounds() -> np.ndarray:
    """
    return domain of branin-hoo function
    """
    return np.array([[-5.0, 10.0],
                     [0.0, 15.0]], dtype=float
                     )

def branin_global_minimum()->float:
    """
    return global minimum of branin hoo function
    """
    return 0.39788735772973816

