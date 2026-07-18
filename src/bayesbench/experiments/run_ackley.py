"""
run experiments on ackley objective function using random_search method
"""
from bayesbench.benchmarks.ackley import ackley, ackley_bounds, ackley_global_minimum
from bayesbench.optimizers.random_search import random_search
from bayesbench.optimizers.gp_ei import gp_expected_improvement
from pathlib import Path
import numpy as np
import pandas as pd


def best_so_far(y: np.ndarray) -> float:
    """
    return minimum in y output so far
    """
    return np.minimum.accumulate(y)

def make_results_df(x: np.ndarray, y: np.ndarray, optimizer: str)->pd.DataFrame:
    """
    create DataFrame object including x, y, steps, optimizer
    """
    df = pd.DataFrame(x, columns = [f"x{i+1}" for i in range(x.shape[1])])
    df.insert(0, "step", np.arange(1, len(y)+1))
    df["objective"]=y
    df["best_so_far"]=best_so_far(y)
    df["optimizer"]=optimizer
    return df

def main():
    d = 2
    bounds = ackley_bounds(d=d)
    print(bounds)
    budget = 50
    seeds = [42, 0, 456]  # Example seeds for reproducibility
    seed = seeds[0]
    #sample inputs and get outputs on objective and return Lists
    rng = np.random.default_rng(seed=seed)
    X, y = random_search(objective=ackley, bounds=bounds, budget=budget, rng=rng)
    df_random = make_results_df(X, y, optimizer="random_search")

    results_ackley_dir = Path("results/ackley")
    results_ackley_dir.mkdir(exist_ok=True)

    output_path = results_ackley_dir / f"ackley_random_search_{seed}.csv"
    df_random.to_csv(output_path, index=False)

    print()
    print(f"Saved results to: {output_path}")
    print(f"Best value found: {df_random['best_so_far'].iloc[-1]:.6f}")

    X, y = gp_expected_improvement(objective=ackley, bounds= bounds, 
                                   budget=budget, rng=rng, random_state=seed)
    df_gp_ei = make_results_df(X, y, optimizer="gp_ei")

    out_put_path = results_ackley_dir / f"ackley_gp_ei_{seed}.csv"
    df_gp_ei.to_csv(out_put_path, index=False)
    print()
    print(f"Saved results to: {output_path}")
    print(f"Best value found: {df_gp_ei['best_so_far'].iloc[-1]:.6f}")
if __name__ == "__main__":
    main()