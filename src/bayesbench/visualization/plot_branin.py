from pathlib import Path

from bayesbench.visualization.plot import plot_best_so_far


def main() -> None:
    input_path = "results/branin/branin_gp_ei.csv"
    output_path = "results/branin/branin_gp_ei.png"

    plot_best_so_far(
        input_path=input_path,
        output_path=output_path,
        title="Branin Optimization: Gaussian Process Expected Improvement",
    )

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()