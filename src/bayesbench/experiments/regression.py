from pathlib import Path

import numpy as np
import pandas as pd

from bayesbench.benchmarks.branin import branin_global_minimum

def make_regret_csv(input_path: str, output_path: str, minimum_value: float)-> None:
    """
    create a csv file with regret values from an experiment result CSV.
    """
    df = pd.read_csv(input_path)
    required = {"step", "objective", "optimizer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")
    df["simple_regret"] = df["best_so_far"] - minimum_value
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    input_path = "results/branin/branin_gp_ei.csv"
    output_path = "results/branin/branin_gp_ei_regret.csv"
    minimum_value = branin_global_minimum()
    make_regret_csv(input_path, output_path, minimum_value)
    print(f"Saved regret CSV to: {output_path}")

if __name__ == "__main__":
    main()