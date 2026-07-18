"""
run experiments on branin objective function using random_search method
"""
from bayesbench.benchmarks.branin import branin, branin_bounds
from bayesbench.experiments.run_experiments import do_experiments
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    bounds = branin_bounds()
    budget = 50
    seeds = [42, 0, 456]  # Example seeds for reproducibility
    seed = seeds[1]
    do_experiments(bounds, budget, seed, branin)
if __name__ == "__main__":
    main()