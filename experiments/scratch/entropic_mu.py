"""
Esperimento base sul solver entropico: prende il grafo descritto in
configs/optimization_parameters.yaml (sezione `graph`), calcola la delta
di Gromov esatta su TUTTE le C(n,4) quadruple, risolve

    max_mu  <mu, delta>  -  T * sum_i mu_i log(mu_i)      mu nel simplesso

con solve_entropic_regularization (mu = softmax(delta / T)), mostra
l'istogramma delle mu e valuta la cost function con quella mu
(entropic_objective in src/optimization/solver.py).

Il grafo si cambia dallo yaml, come in delta_histogram.py: valgono anche
cora/citeseer/pubmed e mutag/proteins/...

    python experiments/scratch/entropic_mu.py
"""

import itertools
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets')))

from src.graphs.utils import create_graph
from src.graphs.visualization import draw_graphs, draw_layout, plot_hist
from src.optimization.objectives import compute_distance_nodes, gromov_energy
from src.optimization.solver import (entropic_objective, entropic_optimum,
                                     solve_entropic_regularization)
from src.utils.config import REPO_ROOT, load_config

from common import TU_DATASETS, citation_patch, tu_graph

# ----------------------------- config ---------------------------------
TEMPERATURES = None   # None = usa optimization.temperature dello yaml;
                      # oppure una lista, es. [0.01, 0.1, 1, 10]
TOP_QUADS = 5         # quante quadruple a mu piu' alta stampare/disegnare
DRAW_TOP = True       # disegnare le TOP_QUADS quadruple sul grafo
SAVE_DIR = None       # es. 'data/scratch' (relativo alla repo)
# ----------------------------------------------------------------------

cfg = load_config()
graph_cfg = cfg["graph"]

if TEMPERATURES is None:
    TEMPERATURES = [cfg["optimization"]["temperature"]]

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
G.remove_edge(4,5)
nodes = list(G.nodes())
dist_matrix, index = compute_distance_nodes(G)

# le quadruple sono in "index space": posizioni in list(G.nodes())
quads = list(itertools.combinations(range(len(nodes)), 4))
deltas = gromov_energy(quads, dist_matrix)

print(f"Grafo: {graph_cfg['type']}   n={len(nodes)}   |E|={G.number_of_edges()}")
print(f"Quadruple totali: {len(quads)}")
print(f"delta max = {deltas.max():g}   delta media = {deltas.mean():.4f}")

if SAVE_DIR is not None:
    save_dir = os.path.join(REPO_ROOT, SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
else:
    save_dir = None

uniform_value = deltas.mean()   # <mu, delta> con mu uniforme, per confronto

for T in TEMPERATURES:
    mu = solve_entropic_regularization(deltas, T)

    weighted_cost = mu @ deltas                  # <mu, delta>
    entropy = -np.sum(mu * np.log(np.clip(mu, 1e-16, None)))
    value = entropic_objective(mu, deltas, T)    # <mu, delta> + T * entropia
    optimum = entropic_optimum(deltas, T)        # T * logsumexp(delta / T)

    print(f"\n{'='*60}\nT = {T}\n{'='*60}")
    print(f"mu: min={mu.min():.3e}  max={mu.max():.3e}  "
          f"(uniforme = {1/len(quads):.3e})")
    print(f"<mu, delta>      = {weighted_cost:.6f}   "
          f"(mu uniforme: {uniform_value:.6f}, delta max: {deltas.max():g})")
    print(f"entropia di mu   = {entropy:.6f}   (max = log|H| = {np.log(len(quads)):.6f})")
    print(f"F(mu) = <mu,delta> + T*entropia = {value:.6f}")
    print(f"valore ottimo T*logsumexp(delta/T) = {optimum:.6f}")

    order = np.argsort(mu)[::-1]
    print(f"\nTop {TOP_QUADS} quadruple per mu:")
    for i in order[:TOP_QUADS]:
        print(f"   {tuple(nodes[j] for j in quads[i])}   "
              f"mu={mu[i]:.3e}   delta={deltas[i]:g}")

    mu_path = os.path.join(save_dir, f"mu_T{T}_{graph_cfg['type']}.png") if save_dir else None
    plot_hist(
        mu,
        title=(f"$\\mu$ = softmax($\\delta$/T) su tutte le quadruple\n"
               f"{graph_cfg['type']} (n={len(nodes)}, |H|={len(quads)}), T={T}\n"
               f"$\\langle\\mu,\\delta\\rangle$={weighted_cost:.4f}, "
               f"F($\\mu$)={value:.4f}, max $\\mu$={mu.max():.2e}"),
        xlabel="$\\mu_i$",
        ylabel="numero di quadruple (scala log)",
        save_path=mu_path,
    )

    if DRAW_TOP:
        to_draw = order[:TOP_QUADS]
        titles = [f"{tuple(nodes[j] for j in quads[i])}\n"
                  f"$\\mu$={mu[i]:.2e}, $\\delta$={deltas[i]:g}" for i in to_draw]
        highlight_nodes = [[nodes[j] for j in quads[i]] for i in to_draw]
        fig, axes = draw_graphs([G] * len(to_draw), [pos] * len(to_draw),
                                titles=titles, highlight_nodes=highlight_nodes)
        if save_dir:
            fig.savefig(os.path.join(save_dir, f"top_quads_T{T}_{graph_cfg['type']}.png"),
                        dpi=150, bbox_inches='tight')
