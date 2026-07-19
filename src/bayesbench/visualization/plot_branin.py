from pathlib import Path

from bayesbench.visualization.plot import plot_random_search, plot_gp_ei, plot_gp_lcb, plot_random_gp_ei_lcb

def plot_branin_random(seeds):
    plot_random_search(seeds=seeds, objective_str="branin")

def plot_branin_gp_ei(seeds):
    plot_gp_ei(seeds=seeds, objective_str="branin")

def plot_branin_gp_lcb(seeds):
    plot_gp_lcb(seeds=seeds, objective_str="branin")

def plot_branin_multi(seeds):
    plot_random_gp_ei_lcb(seeds=seeds, objective_str="branin") 
