"""
run experiments on ackley objective function using random_search method
"""
from bayesbench.benchmarks.ackley import ackley, ackley_bounds
from bayesbench.experiments.run_experiments import run_random_search, run_gp_ei, run_gp_lcb
from pathlib import Path
import numpy as np
import pandas as pd

def run_ackley_random(budget: int, seeds: list[int], d: int, output_dir: str):
    bounds = ackley_bounds(d)
    objective = ackley
    run_random_search(budget=budget, seeds=seeds, bounds=bounds, 
                      objective=objective, output_dir=output_dir)

def run_ackley_gp_ei(budget: int, seeds: list[int], d: int, xi: float, 
                     output_dir: str, n_initial: int, n_candidates: int):
    bounds = ackley_bounds(d)
    objective = ackley
    run_gp_ei(budget=budget, seeds=seeds, bounds=bounds, xi=xi, objective=objective, 
              output_dir=output_dir, n_initial=n_initial, n_candidates=n_candidates)

def run_ackley_gp_lcb(budget: int, seeds: list[int], d: int, beta: float, output_dir: str, n_initial: int, n_candidates: int):
    bounds = ackley_bounds(d)
    objective = ackley
    run_gp_lcb(budget=budget, seeds=seeds, bounds=bounds, beta=beta, objective=objective, 
               output_dir=output_dir, n_initial=n_initial, n_candidates=n_candidates)

def main():
    d = 2
    xi=0.01
    beta = 2
    budget = 50
    seeds = [42, 0, 456]  # Example seeds for reproducibility
    run_ackley_random(budget, seeds, d, output_dir="results/ackley/random")
    run_ackley_gp_ei(budget, seeds, d, xi, output_dir="results/ackley/gp_ei")
    run_ackley_gp_lcb(budget, seeds, d, beta, output_dir="results/ackley/gp_lcb")

if __name__ == "__main__":
    main()