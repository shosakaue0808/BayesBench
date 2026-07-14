"""
run experiments on branin objective function using random_search method
"""
from bayesbench.benchmarks.branin import branin, branin_bounds, branin_global_minimum
from bayesbench.optimizers.random_search import random_search
from bayesbench.optimizers.gp_ei import gp_expected_improvement
from pathlib import Path
import numpy as np
import pandas as pd


def best_so_far(y: np.ndarray) -> np.ndarray:
    """
    Return the minimum objective value found up to each step.

    This assumes minimization.
    """
    return np.minimum.accumulate(y)

def make_results_df(X: np.ndarray, y: np.ndarray, optimizer: str) -> pd.DataFrame:
    """
    create a pandas dataframe to store the results of the optimization run
    """
    return pd.DataFrame(
        {
            "step": np.arange(1, len(y) + 1),
            "x1": X[:, 0],
            "x2": X[:, 1],
            "objective": y,
            "best_so_far": best_so_far(y),
            "optimizer": optimizer,
        }
    )

def main():
    bounds = branin_bounds()
    budget = 50
    
    #sample inputs and get outputs on objective and return Lists
    X, y = random_search(objective=branin, bounds=bounds, budget=budget)
    df_random = make_results_df(X, y, optimizer="random_search")

    results_branin_dir = Path("results/branin")
    results_branin_dir.mkdir(exist_ok=True)

    output_path = results_branin_dir / "branin_random_search.csv"
    df_random.to_csv(output_path, index=False)

    print()
    print(f"Saved results to: {output_path}")
    print(f"Best value found: {df_random['best_so_far'].iloc[-1]:.6f}")

    X, y = gp_expected_improvement(objective=branin, bounds= bounds, budget= budget)
    df_gp_ei = make_results_df(X, y, optimizer="gp_ei")

    out_put_path = results_branin_dir / "branin_gp_ei.csv"
    df_gp_ei.to_csv(out_put_path, index=False)
    print()
    print(f"Saved results to: {output_path}")
    print(f"Best value found: {df_gp_ei['best_so_far'].iloc[-1]:.6f}")
if __name__ == "__main__":
    main()