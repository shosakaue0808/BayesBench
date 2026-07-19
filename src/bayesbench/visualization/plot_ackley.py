from pathlib import Path

from bayesbench.visualization.plot import plot_all


def main() -> None:
    seed = 456
    plot_all(seed, "ackley")
if __name__ == "__main__":
    main()