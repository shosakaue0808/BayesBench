"""
run experiments on branin objective function using random_search method
"""
from collections.abc import Callable
from bayesbench.optimizers.random_search import random_search
from bayesbench.optimizers.gp_ei import gp_expected_improvement
from bayesbench.optimizers.gp_lcb import gp_lcb
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
    df = pd.DataFrame(X, columns = [f"x{i+1}" for i in range(X.shape[1])])
    df.insert(0, "step", np.arange(1, len(y)+1))
    df["objective"]=y
    df["best_so_far"]=best_so_far(y)
    df["optimizer"]=optimizer
    return df

def do_experiments(bounds: np.ndarray, budget: int, seed: int, objective: Callable[[np.ndarray], float]):
    #sample inputs and get outputs on objective and return Lists
    name = objective.__name__
    rng = np.random.default_rng(seed=seed)

    # ---- random_search ----
    results_dir = Path(f"results/{name}/random/")
    results_dir.mkdir(parents=True, exist_ok=True)
    X, y = random_search(objective=objective, bounds=bounds, budget=budget, rng=rng)
    df_random = make_results_df(X, y, optimizer="random_search")
    output_path = results_dir / f"{seed}.csv"
    df_random.to_csv(output_path, index=False)

    print()
    print(f"Saved results to: {output_path}")
    print(f"Best value found: {df_random['best_so_far'].iloc[-1]:.6f}")

    # ---- gp_ei ----
    results_dir = Path(f"results/{name}/gp_ei")
    results_dir.mkdir(parents=True, exist_ok=True)
    X, y = gp_expected_improvement(objective=objective, bounds= bounds, 
                                   budget=budget, rng=rng, random_state=seed)
    df_gp_ei = make_results_df(X, y, optimizer="gp_ei")
    output_path = results_dir / f"{seed}.csv"
    df_gp_ei.to_csv(output_path, index=False)

    print()
    print(f"Saved results to: {output_path}")
    print(f"Best value found: {df_gp_ei['best_so_far'].iloc[-1]:.6f}")

    # ---- gp_lcb ----
    results_dir = Path(f"results/{name}/gp_lcb")  
    results_dir.mkdir(parents=True, exist_ok=True)

    X, y = gp_lcb(objective=objective, bounds=bounds, budget=budget, rng=rng, random_state=seed)
    df_gp_lcb = make_results_df(X, y, optimizer="gp_lcb")
    output_path = results_dir / f"{seed}.csv"
    df_gp_lcb.to_csv(output_path, index=False)

    print(f"Saved results to: {output_path}")
    print(f"Best value found: {df_gp_lcb['best_so_far'].iloc[-1]:.6f}")