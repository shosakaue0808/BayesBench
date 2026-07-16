"""
This module contains functions for visualizing results of csv files in results folder.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {"step", "best_so_far", "optimizer"}
REGRET_COLUMNS = {"step","simple_regret", "optimizer"}

def plot_metric(input_path: str, output_path: str, title: str, columns: set) -> None:
    """
    Plot metric from an experiment result CSV.

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
    output_path = Path(output_path)

    df = pd.read_csv(input_path)

    missing_columns = columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    fig, ax = plt.subplots()

    for optimizer_name, group in df.groupby("optimizer"):
        ax.plot(
            group["step"],
            group[list(columns - {"step", "optimizer"})[0]],
            marker="o",
            label=optimizer_name,
        )

    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel(list(columns - {"step", "optimizer"})[0])
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

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
    plot_metric(input_path, output_path, title, REQUIRED_COLUMNS)

def plot_regret(input_path: str, output_path: str, title: str) -> None:
    """
    Plot regret from an experiment result CSV.

    The input CSV must contain:
        - step
        - simple_regret
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
    plot_metric(input_path, output_path, title, REGRET_COLUMNS)