"""
This module contains functions for visualizing results of csv files in results folder.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = {"step", "best_so_far", "optimizer"}
def plot_best_so_far(input_path: str, output_path: str, title: str) -> None:
    """
    Plot best objective value found so far from an experiment result CSV.

    The input CSV must contain:
        - step
        - best_so_far
        - optimizer

    Parameters
    ----------
    input_path:
        Path to the experiment result CSV.
    output_path:
        Path where the plot image should be saved.
    title:
        Plot title.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    fig, ax = plt.subplots()

    for optimizer_name, group in df.groupby("optimizer"):
        ax.plot(
            group["step"],
            group["best_so_far"],
            marker="o",
            label=optimizer_name,
        )

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Best objective value so far")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)