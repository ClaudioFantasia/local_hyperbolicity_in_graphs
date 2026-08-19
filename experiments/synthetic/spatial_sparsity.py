"""
Which nodes drive the local hyperbolicity of a given target node?

For each node v in optimization.target_nodes we build the k-hop
neighborhood, enumerate/sample its 4-tuples, solve for the optimal
distribution mu over those quads (same KL formulation as
run_experiment.py / KL_score), and then push each quad's mass back onto
its four nodes:

    contribution(j) = sum_{i : j in h_i} mu_i

Nodes sitting in many high-mu quads light up -- those are the ones
responsible for v's G*(v) value. Everything is read from
configs/optimization_parameters.yaml, like run_experiment.py.

    python experiments/synthetic/spatial_sparsity.py
"""

import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_graph
from src.graphs.visualization import draw_graph_with_values, plot_hist
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
    save_dir = os.path.join(REPO_ROOT, save_dir, graph_cfg["type"], "spatial_sparsity")
    os.makedirs(save_dir, exist_ok=True)

print(f"Graph: {graph_cfg['type']}  |  k={k}  temperature={temperature}  "
      f"geometric_temperature={geometric_temperature}  strategy={strategy}")

G, pos = create_graph(**graph_cfg)
nodes = list(G.nodes())
dist_matrix, index = compute_distance_nodes(G)
n_nodes = len(nodes)

tag = f"k{k}_T{temperature}_Tgeom{geometric_temperature}"

# same whole-graph G*(v) heatmap as run_experiment.py, for context:
# the per-target contribution maps below explain one node of this picture
t0 = time.perf_counter()
quad_cache = {}
scores = np.array([
    KL_score(G, index[v], quad_cache, k, temperature, geometric_temperature,
             dist_matrix, strategy, m)
    for v in nodes
])
if scores.ndim == 2 and scores.shape[1] == 1:
    scores = scores[:, 0]
print(f"Scored {n_nodes} nodes in {time.perf_counter() - t0:.2f}s")

if run["plot"]:
    scores_path = os.path.join(save_dir, f"heatmap_{tag}.png") if save_dir else None
    draw_graph_with_values(
        G, pos, scores,
        title=(f"Local hyperbolicity $G^*(v)$ -- {graph_cfg['type']} "
               f"(n={G.number_of_nodes()}, |E|={G.number_of_edges()})\n"
               f"k={k}, $\\beta$={temperature}, $T_{{geom}}$={geometric_temperature}, "
               f"strategy={strategy}\n"
               f"range [{scores.min():.3f}, {scores.max():.3f}], "
               f"mean={scores.mean():.3f}"),
        save_path=scores_path,
    )

for target in opt["target_nodes"]:
    t0 = time.perf_counter()

    neighborhood = get_neighborhood(G, index[target], k, strategy=strategy, m=m)
    quads = sampling_quads(neighborhood)
    deltas = gromov_energy(quads, dist_matrix)
    gamma = gamma_distribution(quads, dist_matrix, index[target],
                               geometric_temperature)
    mu = solve_KL_regularization(deltas, gamma, temperature)

    # G*(v) = <mu, delta> - beta * KL(mu || gamma), the value mu attains
    score = mu @ deltas - temperature * np.sum(
        mu * np.log(np.clip(mu, 1e-16, None) / np.clip(gamma, 1e-16, None)))

    # spread each quad's mass over its four nodes:
    #   contribution[j] = sum_{i : j in h_i} mu_i
    contribution = np.zeros(n_nodes)
    for i, quad in enumerate(quads):
        for node in quad:
            contribution[node] += mu[i]

    elapsed = time.perf_counter() - t0
    print(f"\nnode {target}: {len(neighborhood)}-node {k}-hop, {len(quads)} quads, "
          f"G*={score:.3f}  ({elapsed:.2f}s)")

    order = np.argsort(contribution)[::-1]
    print("  top contributors (node: mu mass):")
    for i in order[:10]:
        print(f"    {nodes[i]}: {contribution[i]:.4f}")

    if not run["plot"]:
        continue

    heat_path = os.path.join(save_dir, f"contrib_node{target}_{tag}.png") if save_dir else None
    draw_graph_with_values(
        G, pos, contribution,
        title=(f"$\\mu$ mass per node -- target {target} ({graph_cfg['type']})\n"
               f"k={k}, $\\beta$={temperature}, $T_{{geom}}$={geometric_temperature}, "
               f"strategy={strategy}\n"
               f"|H|={len(quads)} quads, $G^*$={score:.3f}, "
               f"top node {nodes[order[0]]} ({contribution[order[0]]:.3f})"),
        save_path=heat_path,
    )

    contrib_path = os.path.join(save_dir, f"hist_node{target}_{tag}.png") if save_dir else None
    plot_hist(
        contribution,
        title=(f"Node contributions -- target {target} ({graph_cfg['type']})\n"
               f"mean={contribution.mean():.2e}, max={contribution.max():.2e}"),
        xlabel="$\\sum_{i : j \\in h_i} \\mu_i$",
        ylabel="number of nodes (log scale)",
        bins=30,
        save_path=contrib_path,
    )

if save_dir:
    print(f"\nFigures saved in {save_dir}")
