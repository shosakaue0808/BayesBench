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

def run_random_search(budget: int, seeds: list[int], bounds: np.ndarray, objective: Callable[[np.ndarray], float], output_dir: str):
     # ---- random_search ----
    for seed in seeds:
        rng = np.random.default_rng(seed = seed)
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        X, y = random_search(objective=objective, bounds=bounds, budget=budget, rng=rng)
        df_random = make_results_df(X, y, optimizer="random_search")
        output_path = output_dir / f"{seed}.csv"
        df_random.to_csv(output_path, index=False)

        print()
        print(f"Saved results to: {output_path}")
        print(f"Best value found: {df_random['best_so_far'].iloc[-1]:.6f}")

def run_gp_ei(budget: int, seeds: list[int], bounds: np.ndarray, xi: float,
               objective: Callable[[np.ndarray], float], n_initial: int, n_candidates: int, output_dir: str)->None:
    for seed in seeds:
        rng = np.random.default_rng(seed = seed)
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        X, y = gp_expected_improvement(objective=objective, bounds= bounds, 
                                   budget=budget, rng=rng, xi=xi, random_state=seed, n_initial=n_initial, n_candidates=n_candidates)
        df_gp_ei = make_results_df(X, y, optimizer="gp_ei")
        output_path = output_dir / f"{seed}.csv"
        df_gp_ei.to_csv(output_path, index=False)

        print()
        print(f"Saved results to: {output_path}")
        print(f"Best value found: {df_gp_ei['best_so_far'].iloc[-1]:.6f}")

def run_gp_lcb(budget: int, seeds: list[int], bounds: np.ndarray, beta: float,  
               objective: Callable[[np.ndarray], float], output_dir: str, n_initial: int, n_candidates: int):
    for seed in seeds:
        rng = np.random.default_rng(seed = seed)
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        X, y = gp_lcb(objective=objective, bounds=bounds, budget=budget, beta=beta, 
                      rng=rng, random_state=seed, n_initial=n_initial, n_candidates=n_candidates)
        df_gp_lcb = make_results_df(X, y, optimizer="gp_lcb")
        output_path = output_dir / f"{seed}.csv"
        df_gp_lcb.to_csv(output_path, index=False)

        print(f"Saved results to: {output_path}")
        print(f"Best value found: {df_gp_lcb['best_so_far'].iloc[-1]:.6f}")


def run_all(budget: int, seeds: list[int], bounds: np.ndarray, 
            objective: Callable[[np.ndarray], float], 
            xi: float, beta: float, output_dir_random: str, output_dir_gp_ei: str, output_dir_gp_lcb: str)->None:
    
    run_random_search(budget=budget, seeds=seeds, bounds=bounds, objective=objective, output_dir=output_dir_random)
    run_gp_ei(budget=budget, seeds=seeds, bounds=bounds, xi=xi, objective=objective, output_dir=output_dir_gp_ei)
    run_gp_lcb(budget=budget, seeds=seeds, bounds=bounds, beta=beta, objective=objective, output_dir=output_dir_gp_lcb)
    