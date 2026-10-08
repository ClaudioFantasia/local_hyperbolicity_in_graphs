"""
Confronto della distribuzione di delta su piu' grafi toy: per ognuno
calcola la delta esatta su tutte le C(n,4) quadruple e mette gli
istogrammi affiancati, piu' una tabella riassuntiva.

    python experiments/scratch/compare_graph_types.py
"""

import itertools
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_graph
from src.optimization.objectives import compute_distance_nodes, gromov_energy

# ----------------------------- config ---------------------------------
GRAPHS = [
    {'type': 'tree', 'leaves_per_node': 2, 'tree_height': 3},
    {'type': 'star', 'n': 15},
    {'type': 'path', 'n': 15},
    {'type': 'cycle', 'n': 15},
    {'type': 'lattice', 'n': 4, 'm': 4},
    {'type': 'erdos_renyi', 'n': 15, 'p': 0.3},
]
LOG_SCALE = True
SAVE_PATH = None      # es. 'data/scratch/compare.png' (relativo alla repo)
# ----------------------------------------------------------------------

n_cols = min(len(GRAPHS), 3)
n_rows = math.ceil(len(GRAPHS) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = np.array(axes).flatten()

print(f"{'grafo':<22} {'n':>4} {'|E|':>5} {'quads':>8} {'delta max':>10} "
      f"{'# a delta max':>14} {'delta media':>12}")

for ax, graph_cfg in zip(axes, GRAPHS):
    G, _ = create_graph(**graph_cfg)
    nodes = list(G.nodes())
    dist_matrix, index = compute_distance_nodes(G)

    quads = list(itertools.combinations(range(len(nodes)), 4))
    deltas = gromov_energy(quads, dist_matrix)

    n_max = int(np.sum(np.isclose(deltas, deltas.max())))
    print(f"{graph_cfg['type']:<22} {len(nodes):>4} {G.number_of_edges():>5} "
          f"{len(quads):>8} {deltas.max():>10g} {n_max:>14} {deltas.mean():>12.4f}")

    values, counts = np.unique(deltas, return_counts=True)
    bars = ax.bar([f"{v:g}" for v in values], counts,
                  color='skyblue', edgecolor='black')
    ax.bar_label(bars, fontsize=8, padding=2)
    if LOG_SCALE:
        ax.set_yscale('log')
    ax.set_title(f"{graph_cfg['type']} (n={len(nodes)}, |E|={G.number_of_edges()})\n"
                 f"$\\delta_{{max}}$={deltas.max():g}, media={deltas.mean():.3f}")
    ax.set_xlabel("$\\delta$")
    ax.set_ylabel("numero di quadruple")
    ax.grid(axis='y', alpha=0.3)

for ax in axes[len(GRAPHS):]:
    ax.set_visible(False)

plt.tight_layout()
if SAVE_PATH is not None:
    path = os.path.join(os.path.dirname(__file__), '../..', SAVE_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
plt.show()
