from pathlib import Path

from bayesbench.visualization.plot import plot_all

def main() -> None:
    seed = 42
    plot_all(seed, "branin")
if __name__ == "__main__":
    main()