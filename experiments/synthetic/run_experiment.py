"""
Compute local (KL-divergence) Gromov hyperbolicity for every node of a
toy graph and visualize the result as a heatmap.

Everything is set in configs/optimization_parameters.yaml:
  graph:        which graph to build (type + its shape params). `type` can
                also be a real dataset instead of a synthetic graph:
                  cora/citeseer/pubmed -- a small connected piece of the
                    real graph (BFS from `source_node`, or from a random
                    node if it is null, first `n` nodes): same degree
                    structure as a citation network, small enough to
                    score exactly.
                  mutag/proteins/... -- graph number `graph_index` of a TU
                    benchmark, taken whole (a MUTAG molecule is 10-28
                    nodes, so nothing has to be cut).
  optimization: method (KL_divergence or entropic), k, temperature (beta),
                geometric_temperature (KL_divergence) / lambda_loc (entropic),
                strategy, m
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
from src.optimization.local import KL_score, tempo_KL_score, entropic_score
from src.optimization.neighborhood import get_neighborhood, sampling_quads
from src.optimization.objectives import (compute_distance_nodes, gamma_distribution,
                                         gromov_energy, tuple_distance)
from src.optimization.solver import (solve_KL_regularization,
                                     solve_local_entropic_regularization)
from src.utils.config import REPO_ROOT, load_config

from common import TU_DATASETS, citation_patch, tu_graph

cfg = load_config()
graph_cfg = cfg["graph"]
opt = cfg["optimization"]
run = cfg["run"]

method = opt["method"]
k = opt["k"]
temperature = opt["temperature"]
geometric_temperature = opt["geometric_temperature"]
lambda_loc = opt["lambda_loc"]
strategy = opt["strategy"]
m = opt["m"]

# TEMPORARY: True -> score every node with <mu, delta> (tempo_KL_score, the
# reward term alone), False -> the full V* of KL_score. KL_divergence only.
plot_mu_delta = False

save_dir = run["save_dir"]
if save_dir is not None:
    save_dir = os.path.join(REPO_ROOT, save_dir, graph_cfg["type"])
    os.makedirs(save_dir, exist_ok=True)

print(f"Graph: {graph_cfg['type']}  |  method={method}  k={k}  "
      f"temperature={temperature}  geometric_temperature={geometric_temperature}  "
      f"lambda_loc={lambda_loc}  strategy={strategy}")

if graph_cfg["type"] in ("cora", "citeseer", "pubmed"):
    G = citation_patch(graph_cfg["type"], k=graph_cfg["patch_k"],
                       source_node=graph_cfg.get("source_node"),
                       seed=graph_cfg.get("seed", 0),
                       strategy=graph_cfg.get("patch_strategy", "full_neighborhood"),
                       m=graph_cfg.get("patch_m", 5))
    pos = draw_layout(G, seed=42)
elif graph_cfg["type"] in TU_DATASETS:
    G = tu_graph(graph_cfg["type"], graph_cfg.get("graph_index", 0))
    pos = draw_layout(G, seed=42)
else:
    G, pos = create_graph(**graph_cfg)

# TEMPORARY: star (centre 0) with "double triangles" hardcoded on top.
# star_variant = 1 -> one double triangle: leaves 1-2-3 joined by (1,2),(2,3)
# star_variant = 2 -> same, plus m_leaves new plain leaves and a second
#                     identical double triangle made of 3 new leaves
star_variant = 2
m_leaves = 3
if graph_cfg["type"] == "star":
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    if star_variant == 2:
        new = G.number_of_nodes()
        for i in range(m_leaves):
            G.add_edge(0, new + i)
        a, b, c = new + m_leaves, new + m_leaves + 1, new + m_leaves + 2
        G.add_edge(0, a)
        G.add_edge(0, b)
        G.add_edge(0, c)
        G.add_edge(a, b)
        G.add_edge(b, c)
    pos = draw_layout(G, seed=42)


# same order as compute_distance_nodes' index, so scores[index[v]] is v's score
nodes = list(G.nodes())
dist_matrix, index = compute_distance_nodes(G)
quad_cache = {}

t0 = time.perf_counter()
if method == "entropic":
    # cost = delta(h) - lambda_loc*d_v(h): the locality is in the cost,
    # so lambda_loc replaces T_geom (lambda_loc = temperature / T_geom)
    scores = np.array([
        entropic_score(G, index[v], quad_cache, k, temperature, lambda_loc,
                       dist_matrix, strategy, m)
        for v in nodes
    ])
else:
    score_fn = tempo_KL_score if plot_mu_delta else KL_score
    scores = np.array([
        score_fn(G, index[v], quad_cache, k, temperature, geometric_temperature,
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
    if method == "entropic":
        tag = f"entropic_k{k}_T{temperature}_lambda{lambda_loc}"
        params = (f"k={k}, $\\beta$={temperature}, $\\lambda$={lambda_loc}, "
                  f"strategy={strategy}")
    else:
        tag = f"KL_k{k}_T{temperature}_Tgeom{geometric_temperature}"
        params = (f"k={k}, $\\beta$={temperature}, "
                  f"$T_{{geom}}$={geometric_temperature}, strategy={strategy}")
    if plot_mu_delta and method != "entropic":
        tag = "mudelta_" + tag
    if graph_cfg["type"] == "star":
        tag = f"var{star_variant}_" + tag
    score_name = ("$\\langle\\mu, \\delta\\rangle(v)$" if plot_mu_delta and method != "entropic"
                  else "Local hyperbolicity $G^*(v)$")

    heat_path = os.path.join(save_dir, f"heatmap_{tag}.png") if save_dir else None

    draw_graph_with_values(
        G, pos, scores,
        title=(f"{score_name} -- {graph_cfg['type']} "
               f"(n={G.number_of_nodes()}, |E|={G.number_of_edges()}, {method})\n"
               f"{params}\n"
               f"range [{scores.min():.3f}, {scores.max():.3f}], "
               f"mean={scores.mean():.3f}"),
        save_path=heat_path,
    )

    # the distribution mu over quads behind G*(v), for each reference node
    # in target_nodes: argmax of <mu, delta> - beta*KL(mu || gamma_v) for
    # KL_divergence, of <mu, delta - lambda_loc*d_v> + beta*entropy(mu)
    # for entropic
    for target in opt["target_nodes"]:
        neighborhood = get_neighborhood(G, index[target], k, strategy=strategy, m=m)
        quads = sampling_quads(neighborhood)
        deltas = gromov_energy(quads, dist_matrix)
        if method == "entropic":
            w = tuple_distance(quads, dist_matrix, index[target])
            mu = solve_local_entropic_regularization(deltas, w, temperature,
                                                     lambda_loc)
        else:
            from scipy.special import softmax
            q = np.array(quads)
            print(q)
            w = dist_matrix[q, target].mean(axis=1)
            print(w)
            gamma = softmax(-w / geometric_temperature)
            #gamma = gamma_distribution(quads, dist_matrix, index[target],
            #                           geometric_temperature)
            print(gamma)
            mu = solve_KL_regularization(deltas, gamma, temperature)

        top = np.argmax(mu)

        mu_path = os.path.join(save_dir, f"mu_node{target}_{tag}.png") if save_dir else None
        plot_hist(
            mu,
            title=(f"$\\mu$ over quads -- node {target} ({graph_cfg['type']}, "
                   f"{method}, |H|={len(quads)} quads, "
                   f"{len(neighborhood)}-node {k}-hop)\n"
                   f"{params}, $G^*$={scores[index[target]]:.3f}\n"
                   f"max $\\mu$={mu.max():.2e} at {quads[top]} "
                   f"($\\delta$={deltas[top]:.2f})"),
            xlabel="$\\mu_i$",
            ylabel="number of quads (log scale)",
            save_path=mu_path,
        )

        # TEMPORARY: the reference distribution gamma_v the KL pulls mu towards
        if method != "entropic":
            gamma_path = os.path.join(save_dir, f"gamma_node{target}_{tag}.png") if save_dir else None
            plot_hist(
                gamma,
                title=(f"$\\gamma_v$ over quads -- node {target} ({graph_cfg['type']}, "
                       f"|H|={len(quads)} quads)\n"
                       f"$T_{{geom}}$={geometric_temperature}, "
                       f"max $\\gamma$={gamma.max():.2e}, min $\\gamma$={gamma.min():.2e}"),
                xlabel="$\\gamma_v(h_i)$",
                ylabel="number of quads (log scale)",
                save_path=gamma_path,
            )

    if save_dir:
        print(f"Figures saved in {save_dir}")
