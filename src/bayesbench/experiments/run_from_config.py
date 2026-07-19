import yaml

from pathlib import Path
import argparse
from bayesbench.experiments.run_ackley import run_ackley_random, run_ackley_gp_ei, run_ackley_gp_lcb
from bayesbench.experiments.run_branin import run_branin_random, run_branin_gp_ei, run_branin_gp_lcb
from bayesbench.visualization.plot_branin import plot_branin_random, plot_branin_gp_ei, plot_branin_gp_lcb, plot_branin_multi
from bayesbench.visualization.plot_ackley import plot_ackley_random, plot_ackley_gp_ei, plot_ackley_gp_lcb, plot_ackley_multi


def resolve_config_path(config_name: str) -> Path:
    path = Path(config_name)

    # If user already passed a path like configs/branin_gp_ei.yaml, keep it
    if path.parent != Path("."):
        return path

    # Otherwise treat it as just a file name inside repo_root/configs/
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / path.name

def load_yaml(yaml_path: Path)-> dict:
    if not yaml_path.exists():
        raise FileNotFoundError(f"config file not found: {yaml_path}")
        
    # Open and read the YAML file
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    return data

def main():
    # description is for --help flag
    parser = argparse.ArgumentParser(description="This will process yaml file and implements corresponding experiments")
    parser.add_argument("config", type=str, help="file name of yaml")
    args = parser.parse_args()
    yaml_path = resolve_config_path(args.config)

    data = load_yaml(yaml_path)
    print(data)
    
    benchmark = data["benchmark"]
    optimizer = data["optimizer"]
    optim_params = optimizer["params"]
    run_config = data["run"]
    budget = run_config["budget"]
    seeds = run_config["seeds"]
    output_dir = run_config["output_dir"]

    if(benchmark["name"] == "ackley"):
        dimension = benchmark["params"]["d"]
        if(optimizer["name"]=="all"):
            run_ackley_random(budget, seeds, dimension, output_dir["random"])
            plot_ackley_random(seeds)
            run_ackley_gp_ei(budget=budget, seeds=seeds, d=dimension, xi=optim_params["xi"],
                            output_dir=output_dir["gp_ei"], n_initial=optim_params["n_initial"], n_candidates=optim_params["n_candidates"])
            plot_ackley_gp_ei(seeds)
            run_ackley_gp_lcb(budget=budget, seeds=seeds, d=dimension, beta=optim_params["beta"],
                               output_dir=output_dir["gp_lcb"], n_initial=optim_params["n_initial"], n_candidates=optim_params["n_candidates"])
            plot_ackley_gp_lcb(seeds)
            print("---------------so far------------")
            plot_ackley_multi(seeds)
            
        elif(optimizer["name"] == "random_search"):
            run_ackley_random(budget, seeds, dimension, output_dir)
            plot_ackley_random(seeds)
        elif(optimizer["name"]=="gp_ei"):
            run_ackley_gp_ei(budget=budget, seeds=seeds, d=dimension, xi=optim_params["xi"],
                            output_dir=output_dir, n_initial=optim_params["n_initial"], n_candidates=optim_params["n_candidates"])
            plot_ackley_gp_ei(seeds)
        else:
            run_ackley_gp_lcb(budget=budget, seeds=seeds, d=dimension, beta=optim_params["beta"],
                            output_dir=output_dir, n_initial=optim_params["n_initial"], n_candidates=optim_params["n_candidates"])
            plot_ackley_gp_lcb(seeds)
    else:
        if(optimizer["name"] == "all"):
            run_branin_random(budget, seeds, output_dir["random"])
            plot_branin_random(seeds)
            run_branin_gp_ei(budget, seeds, optim_params["xi"], optim_params["n_initial"], 
                             optim_params["n_candidates"], output_dir["gp_ei"])
            plot_branin_gp_ei(seeds)
            run_branin_gp_lcb(budget, seeds, optim_params["beta"], 
                              optim_params["n_initial"], optim_params["n_candidates"], output_dir["gp_lcb"]) 
            plot_branin_gp_lcb(seeds)
            print("---------------so far------------")
            plot_branin_multi(seeds)

        elif(optimizer["name"] == "random_search"):
            run_branin_random(budget, seeds, output_dir)
            plot_branin_random(seeds)
        elif(optimizer["name"]=="gp_ei"):
            run_branin_gp_ei(budget, seeds, optim_params["xi"], optim_params["n_initial"], 
                            optim_params["n_candidates"], output_dir)
            plot_branin_gp_ei(seeds)
        else:
            run_branin_gp_lcb(budget, seeds, optim_params["beta"], 
                            optim_params["n_initial"], optim_params["n_candidates"], output_dir) 
            plot_branin_gp_lcb(seeds)

if __name__ == "__main__":
    main()


