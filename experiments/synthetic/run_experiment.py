"""
Compute local (KL-divergence) Gromov hyperbolicity for every node of a
toy graph and visualize the result as a heatmap.

Everything is set in configs/optimization_parameters.yaml:
  graph:        which graph to build (type + its shape params). `type` can
                also be cora/citeseer/pubmed, which takes a small connected
                piece of the real graph (BFS from `source_node`, or from a
                random node if it is null, first `n` nodes) instead of a
                synthetic one -- same degree structure as a citation
                network, small enough to score exactly.
  optimization: k, temperature (beta), geometric_temperature, strategy, m
  run:          whether to plot and where to save the figures

    python experiments/synthetic/run_experiment.py
"""

import math
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets')))

from src.graphs.utils import create_graph
from src.graphs.visualization import draw_graph_with_values, draw_layout, plot_hist
from src.optimization.local import KL_score
from src.optimization.neighborhood import get_neighborhood, sampling_quads
from src.optimization.objectives import (compute_distance_nodes, gamma_distribution,
                                         gromov_energy)
from src.optimization.solver import solve_KL_regularization
from src.utils.config import REPO_ROOT, load_config

cfg = load_config()
graph_cfg = cfg["graph"]
opt = cfg["optimization"]
run = cfg["run"]

k = opt["k"]
temperature = opt["temperature"]
geometric_temperature = opt["geometric_temperature"]
strategy = opt["strategy"]
m = opt["m"]

save_dir = run["save_dir"]
if save_dir is not None:
    save_dir = os.path.join(REPO_ROOT, save_dir, graph_cfg["type"])
    os.makedirs(save_dir, exist_ok=True)

print(f"Graph: {graph_cfg['type']}  |  k={k}  temperature={temperature}  "
      f"geometric_temperature={geometric_temperature}  strategy={strategy}")

if graph_cfg["type"] in ("cora", "citeseer", "pubmed"):
    from common import citation_patch
    G = citation_patch(graph_cfg["type"], n=graph_cfg["n"],
                       source_node=graph_cfg.get("source_node"))
    pos = draw_layout(G, seed=42)
else:
    G, pos = create_graph(**graph_cfg)

# same order as compute_distance_nodes' index, so scores[index[v]] is v's score
nodes = list(G.nodes())
dist_matrix, index = compute_distance_nodes(G)
quad_cache = {}

t0 = time.perf_counter()
scores = np.array([
    KL_score(G, index[v], quad_cache, k, temperature, geometric_temperature,
             dist_matrix, strategy, m)
    for v in nodes
])
elapsed = time.perf_counter() - t0

# KL_score returns one value per geometric_temperature; with a single scalar
# T_geom collapse that trailing axis so scores is a plain (n_nodes,) array.
if scores.ndim == 2 and scores.shape[1] == 1:
    scores = scores[:, 0]

print(f"Scored {len(nodes)} nodes in {elapsed:.2f}s  "
      f"(quad cache: {len(quad_cache)} entries, "
      f"theoretical max for exact global score: {math.comb(len(nodes), 4)})")

if run["plot"]:
    tag = f"k{k}_T{temperature}_Tgeom{geometric_temperature}"
    heat_path = os.path.join(save_dir, f"heatmap_{tag}.png") if save_dir else None

    draw_graph_with_values(
        G, pos, scores,
        title=(f"Local hyperbolicity $G^*(v)$ -- {graph_cfg['type']} "
               f"(n={G.number_of_nodes()}, |E|={G.number_of_edges()})\n"
               f"k={k}, $\\beta$={temperature}, $T_{{geom}}$={geometric_temperature}, "
               f"strategy={strategy}\n"
               f"range [{scores.min():.3f}, {scores.max():.3f}], "
               f"mean={scores.mean():.3f}"),
        save_path=heat_path,
    )

    # mu = argmax_mu <mu, delta> - beta*KL(mu || gamma_v), the distribution
    # over quads behind G*(v), for each reference node in target_nodes
    for target in opt["target_nodes"]:
        neighborhood = get_neighborhood(G, index[target], k, strategy=strategy, m=m)
        quads = sampling_quads(neighborhood)
        deltas = gromov_energy(quads, dist_matrix)
        gamma = gamma_distribution(quads, dist_matrix, index[target],
                                   geometric_temperature)
        mu = solve_KL_regularization(deltas, gamma, temperature)

        top = np.argmax(mu)

        mu_path = os.path.join(save_dir, f"mu_node{target}_{tag}.png") if save_dir else None
        plot_hist(
            mu,
            title=(f"$\\mu$ over quads -- node {target} ({graph_cfg['type']}, "
                   f"|H|={len(quads)} quads, {len(neighborhood)}-node {k}-hop)\n"
                   f"$\\beta$={temperature}, $T_{{geom}}$={geometric_temperature}, "
                   f"$G^*$={scores[index[target]]:.3f}\n"
                   f"max $\\mu$={mu.max():.2e} at {quads[top]} "
                   f"($\\delta$={deltas[top]:.2f})"),
            xlabel="$\\mu_i$",
            ylabel="number of quads (log scale)",
            save_path=mu_path,
        )

    if save_dir:
        print(f"Figures saved in {save_dir}")
