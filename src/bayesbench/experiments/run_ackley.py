"""
run experiments on ackley objective function using random_search method
"""
from bayesbench.benchmarks.ackley import ackley, ackley_bounds
from bayesbench.experiments.run_experiments import do_experiments
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    d = 2
    bounds = ackley_bounds(d=d)
    budget = 50
    seeds = [42, 0, 456]  # Example seeds for reproducibility
    seed = seeds[2]
    do_experiments(bounds, budget, seed, ackley)
if __name__ == "__main__":
    main()