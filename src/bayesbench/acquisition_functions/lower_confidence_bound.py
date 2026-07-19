"""
This file implements Lower Confidence Bound (LCB), type of acquisition function used in Bayesian optimization

a(x) = mean(x) - c*sigma(x), where mean is predicted mean of objective function (exploitation, likely to be good), 
sigma is predicted standard diviation or uncertainty (exploration, uncertain area), and c is tuning parameter.

https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html
"""

import numpy as np

def lower_confidence_bound(mean: np.ndarray, stddiv: np.ndarray, beta: float) -> np.ndarray:
    """
    lower confidence bound acquisition function
    - mean is computed mean of GP (sarogate model) at candidates points, smaller is better
    - stddiv is computed standard diviation of GP at candidate points, larger is better
    - param is tunable parameter
    """
    return mean - beta * stddiv