"""
run experiments on branin objective function using random_search method
"""
from bayesbench.benchmarks.branin import branin, branin_bounds, branin_global_minimum
from bayesbench.optimizers.random_search import random_search
from pathlib import Path
import numpy as np
import pandas as pd


def best_so_far(y: np.ndarray) -> np.ndarray:
    """
    Return the minimum objective value found up to each step.

    This assumes minimization.
    """
    return np.minimum.accumulate(y)

def main():
    bounds = branin_bounds()
    budget = 50
    
    #sample inputs and get outputs on objective and return Lists
    X, y = random_search(objective=branin, bounds=bounds, budget=budget)
    df = pd.DataFrame(
        {
            "step": np.arange(1, budget + 1),
            "x1": X[:, 0],
            "x2": X[:, 1],
            "objective": y,
            "best_so_far": best_so_far(y),
            "optimizer": "random_search",
        }
    )

    results_dir = Path("results/branin")
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "branin_random_search.csv"
    df.to_csv(output_path, index=False)

    print(df.tail())
    print()
    print(f"Saved results to: {output_path}")
    print(f"Best value found: {df['best_so_far'].iloc[-1]:.6f}")

if __name__ == "__main__":
    main()