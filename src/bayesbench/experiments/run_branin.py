"""
run experiments on branin objective function using random_search method
"""
from bayesbench.benchmarks.branin import branin, branin_bounds
from bayesbench.experiments.run_experiments import run_random_search, run_gp_ei, run_gp_lcb
from pathlib import Path
import numpy as np
import pandas as pd

def run_branin_random(budget: int, seeds: list[int], output_dir: str):
    bounds = branin_bounds()
    objective = branin
    run_random_search(budget=budget, seeds=seeds, bounds=bounds, objective=objective, output_dir=output_dir)

def run_branin_gp_ei(budget: int, seeds: list[int], xi: float, 
                     n_initial: int, n_candidates: int, output_dir: str):
    bounds = branin_bounds()
    objective = branin
    run_gp_ei(budget=budget, seeds=seeds, bounds=bounds, xi=xi, objective=objective, 
              n_initial=n_initial, n_candidates=n_candidates, output_dir=output_dir)

def run_branin_gp_lcb(budget: int, seeds: list[int], beta: float, 
                      n_initial: int, n_candidates: int, output_dir: str):
    bounds = branin_bounds()
    objective = branin
    run_gp_lcb(budget=budget, seeds=seeds, bounds=bounds, beta=beta, objective=objective,
               n_initial=n_initial, n_candidates=n_candidates, output_dir=output_dir)


def main():
    budget = 50
    seeds = [42, 0, 456]  # Example seeds for reproducibility
    xi=0.01
    beta=2.0
    run_branin_random(budget, seeds, output_dir="results/branin/random")
    run_branin_gp_ei(budget, seeds, xi, output_dir="results/branin/gp_ei")
    run_branin_gp_lcb(budget, seeds, beta, output_dir="results/branin/gp_lcb")

if __name__ == "__main__":
    main()