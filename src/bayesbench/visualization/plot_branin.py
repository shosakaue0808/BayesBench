from pathlib import Path

from bayesbench.visualization.plot import plot_best_so_far, plot_multi_opt_best_so_far, plot_regret


def main() -> None:
    seed = 0
    input_path_gp_ei = f"results/branin/branin_gp_ei_{seed}.csv"
    input_path_random_search = f"results/branin/branin_random_search_{seed}.csv"
    input_paths = [input_path_gp_ei, input_path_random_search]
    output_path = f"results/branin/branin_multi_opt_{seed}.png"

    plot_multi_opt_best_so_far(
        input_paths=input_paths,
        output_path=output_path,
        title=f"Branin Optimization: Multiple Optimizers",
        seed=seed
    )

    # plot_best_so_far(
    #     input_path=input_path,
    #     output_path=output_path,
    #     title="Branin Optimization: Gaussian Process Expected Improvement",
    #     seed=seed
    # )

    # print(f"Saved plot to: {output_path}")

    # input_path = f"results/branin/branin_random_search_{seed}.csv"
    # output_path = f"results/branin/branin_random_search_{seed}.png"

    # plot_best_so_far(
    #     input_path=input_path,
    #     output_path=output_path,
    #     title="Branin Optimization: Random Search",
    #     seed=seed
    # )

    print(f"Saved plot to: {output_path}")

if __name__ == "__main__":
    main()