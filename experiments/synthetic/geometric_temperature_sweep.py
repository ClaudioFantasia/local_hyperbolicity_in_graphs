"""
Effect of the geometric temperature T_geom on the local hyperbolicity
G*(v) = KL_score(v), on the toy graph set in
configs/optimization_parameters.yaml.

beta (the KL regularisation strength) is fixed here, and k is set to
infinity, so the neighborhood of every node is the whole graph and the
only thing that changes across the sweep is gamma_v, the reference
distribution over quads:

    gamma_v(h) propto exp(-d_v(h) / T_geom)

T_geom -> 0   : gamma_v collapses on the quads closest to v, so G*(v)
                only sees the quads sitting on top of v.
T_geom -> inf : gamma_v becomes uniform over all quads of the graph, so
                G*(v) stops depending on v and converges to the same
                (global) value for every node -- with beta small that
                value is the exact global delta.

In between there is a window of T_geom where the curves actually move
and separate node from node: that window is what the script plots and
prints (per node, the T_geom range covering the middle 90% of the
transition from the T_geom->0 plateau to the T_geom->inf one).

    python experiments/synthetic/geometric_temperature_sweep.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets')))

from src.graphs.utils import create_graph
from src.optimization.local import KL_score
from src.optimization.objectives import compute_distance_nodes, compute_gromov_hyperbolicity
from src.utils.config import REPO_ROOT, load_config
from src.graphs.visualization import draw_graphs,draw_layout
from common import TU_DATASETS, citation_patch, tu_graph

# ----------------------------- config ---------------------------------
BETA = 0.1                         # temperature of KL_score, fixed
T_GEOM = np.logspace(-3, 2, 80)        # the sweep
K = 1000                           # k = infinity: the ball is the whole graph
SEED = 0                               # which quads sampling_quads draws, if it has to
# ----------------------------------------------------------------------

cfg = load_config()
graph_cfg = cfg["graph"]
run = cfg["run"]

if graph_cfg["type"] in ("cora", "citeseer", "pubmed"):
    G = citation_patch(graph_cfg["type"], k=graph_cfg["patch_k"],
                       source_node=graph_cfg.get("source_node"),
                       seed=graph_cfg.get("seed", 0),
                       strategy=graph_cfg.get("patch_strategy", "full_neighborhood"),
                       m=graph_cfg.get("patch_m", 5))
elif graph_cfg["type"] in TU_DATASETS:
    G = tu_graph(graph_cfg["type"], graph_cfg.get("graph_index", 0))
else:
    G, _ = create_graph(**graph_cfg)

nodes = list(G.nodes())
dist_matrix, index = compute_distance_nodes(G)
quad_cache = {}

pos = draw_layout(G)
draw_graphs(G,pos)


print(f"Graph: {graph_cfg['type']}   n={len(nodes)}   |E|={G.number_of_edges()}   "
      f"beta={BETA}   k=inf   T_geom in [{T_GEOM[0]:g}, {T_GEOM[-1]:g}] "
      f"({len(T_GEOM)} points)")

# KL_score takes the whole T_geom grid at once: the quads, their distances
# to v and their Gromov energies are computed once per node and reused for
# every T_geom, so this is one pass over the nodes, not len(T_GEOM) passes.
scores = np.array([
    KL_score(G, index[v], quad_cache, K, BETA, T_GEOM, dist_matrix, seed=SEED)
    for v in nodes
])  # shape (n_nodes, len(T_GEOM))

delta_global = compute_gromov_hyperbolicity(G)

# where each curve moves: the T_geom range covering the middle 90% of the
# gap between the two plateaus (5% -> 95% of |G*(inf) - G*(0)|)
print(f"\nexact global delta = {delta_global:.3f}")
print(f"{'node':>6} {'G*(T->0)':>10} {'G*(T->inf)':>12} {'active T_geom range':>24}")
lo_all, hi_all = [], []
for v in nodes:
    curve = scores[index[v]]
    low, high = curve[0], curve[-1]
    span = high - low
    if abs(span) < 1e-9:
        print(f"{v:>6} {low:>10.3f} {high:>12.3f} {'flat (no effect)':>24}")
        continue
    # first/last T_geom where the curve is still within 5% of a plateau
    moving = np.where((curve - low) / span > 0.10)[0]
    settled = np.where((curve - low) / span > 0.90)[0]
    t_lo = T_GEOM[moving[0]]
    t_hi = T_GEOM[settled[0]] if len(settled) else T_GEOM[-1]
    lo_all.append(t_lo)
    hi_all.append(t_hi)
    print(f"{v:>6} {low:>10.3f} {high:>12.3f} {f'[{t_lo:.3g}, {t_hi:.3g}]':>24}")

if lo_all:
    print(f"\nOverall, G*(v) reacts to T_geom in [{min(lo_all):.3g}, {max(hi_all):.3g}]; "
          f"outside it every curve sits on one of its two plateaus.")

if run["plot"]:
    plt.figure(figsize=(9, 6))
    for v in nodes:
        plt.plot(T_GEOM, scores[index[v]], lw=1, alpha=0.5)
    plt.plot(T_GEOM, scores.mean(axis=0), lw=2.5, color="black", label="mean over nodes")
    plt.axhline(delta_global, ls="--", color="gray",
                label=f"exact global $\\delta$ = {delta_global:.2f}")
    if lo_all:
        plt.axvspan(min(lo_all), max(hi_all), color="orange", alpha=0.12,
                    label="range where $T_{geom}$ matters")
    plt.xscale("log")
    plt.xlabel("$T_{geom}$")
    plt.ylabel("$G^*(v)$")
    plt.title(f"Effect of $T_{{geom}}$ on $G^*(v)$ -- {graph_cfg['type']} "
              f"(n={len(nodes)}, |E|={G.number_of_edges()})\n"
              f"$\\beta$={BETA}, k={K}, one line per node")
    plt.legend()
    plt.tight_layout()

    if run["save_dir"] is not None:
        save_dir = os.path.join(REPO_ROOT, run["save_dir"], graph_cfg["type"])
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"Tgeom_sweep_beta{BETA}.png")
        plt.savefig(path, dpi=150)
        print(f"\nFigure saved in {path}")
    plt.show()
