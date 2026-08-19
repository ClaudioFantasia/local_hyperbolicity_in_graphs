"""
Filtration over the neighborhood radius: for every node v, record the
smallest k at which its local hyperbolicity G*(v) first exceeds a
threshold tau. Nodes sitting next to a non-tree-like region light up at
small k, nodes deep inside a tree only at large k (or never, -1).

Graph and scoring hyperparameters come from
configs/optimization_parameters.yaml, like the other scripts here; tau
and the max k swept are set below.

    python experiments/synthetic/filtration.py

Note: local_hyperbolicity_valutazione.md (Sec. 10 / B.6) argues this
filtration should run over T_geom rather than k, since k is only a
computational truncation. This script still sweeps k -- it is the
original experiment, kept as-is.
"""

import os
import sys
import time

import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_graph
from src.graphs.visualization import draw_graph_with_values, plot_hist
from src.optimization.local import KL_score
from src.optimization.objectives import (compute_distance_nodes,
                                         compute_gromov_hyperbolicity)
from src.utils.config import REPO_ROOT, load_config

tau = 0.1  # a node "activates" the first time its G*(v) goes above this
## NOTE that this is the value threshold, so the first time that a leaf node see the support 
## of non-hyperbolic regions

cfg = load_config()
graph_cfg = cfg["graph"]
opt = cfg["optimization"]
run = cfg["run"]

temperature = opt["temperature"]
geometric_temperature = opt["geometric_temperature"]
strategy = opt["strategy"]
m = opt["m"]

save_dir = run["save_dir"]
if save_dir is not None:
    save_dir = os.path.join(REPO_ROOT, save_dir, graph_cfg["type"], "filtration")
    os.makedirs(save_dir, exist_ok=True)

G, pos = create_graph(**graph_cfg)
nodes = list(G.nodes())
dist_matrix, index = compute_distance_nodes(G)
diameter = nx.diameter(G)

print(f"Graph: {graph_cfg['type']}  |  n={G.number_of_nodes()}  "
      f"|E|={G.number_of_edges()}  diameter={diameter}")
print(f"tau={tau}  temperature={temperature}  "
      f"geometric_temperature={geometric_temperature}  strategy={strategy}")

# exact global delta is only tractable on small graphs (comb(n, 4) quads)
if G.number_of_nodes() <= 300:
    print(f"Exact global Gromov hyperbolicity: {compute_gromov_hyperbolicity(G):.3f}")

quad_cache = {}
filtration_scores = np.full(len(nodes), -1.0)
scores = np.zeros(len(nodes))

t0 = time.perf_counter()
for k in range(diameter + 1):
    scores = np.array([
        KL_score(G, index[v], quad_cache, k, temperature, geometric_temperature,
                 dist_matrix, strategy, m)
        for v in nodes
    ])
    # KL_score returns one value per geometric_temperature; with a scalar
    # T_geom collapse that trailing axis
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores[:, 0]

    newly_active = (scores > tau) & (filtration_scores == -1)
    filtration_scores[newly_active] = k
    print(f"  k={k}: {newly_active.sum()} nodes activated, "
          f"{(filtration_scores == -1).sum()} still below tau")

    if np.all(filtration_scores != -1):
        break

print(f"Swept k in {time.perf_counter() - t0:.2f}s "
      f"(quad cache: {len(quad_cache)} entries)")

if run["plot"]:
    tag = f"tau{tau}_T{temperature}_Tgeom{geometric_temperature}"

    heat_path = os.path.join(save_dir, f"heatmap_{tag}.png") if save_dir else None
    draw_graph_with_values(
        G, pos, scores,
        title=(f"Local hyperbolicity $G^*(v)$ at the last k swept -- "
               f"{graph_cfg['type']}\n"
               f"$\\beta$={temperature}, $T_{{geom}}$={geometric_temperature}"),
        save_path=heat_path,
    )

    filt_path = os.path.join(save_dir, f"filtration_{tag}.png") if save_dir else None
    draw_graph_with_values(
        G, pos, filtration_scores,
        title=(f"Filtration: smallest k with $G^*(v) > \\tau$ -- "
               f"{graph_cfg['type']}\n"
               f"$\\tau$={tau}, $\\beta$={temperature}, "
               f"$T_{{geom}}$={geometric_temperature}\n"
               f"(-1 = never activated up to k={diameter})"),
        save_path=filt_path,
    )

    plot_hist(
        scores,
        title=(f"Local hyperbolicity (mean={scores.mean():.4f}, "
               f"var={scores.var():.4f})"),
        xlabel="$G^*(v)$",
        ylabel="number of nodes (log scale)",
        bins=20,
        save_path=os.path.join(save_dir, f"hist_{tag}.png") if save_dir else None,
    )

    if save_dir:
        print(f"Figures saved in {save_dir}")
