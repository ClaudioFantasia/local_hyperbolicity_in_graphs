"""
Esperimento base (ripartenza da zero, stile notebook 01):
prende il grafo descritto in configs/optimization_parameters.yaml
(sezione `graph`), calcola la delta di Gromov esatta su TUTTE le
C(n,4) quadruple, mostra l'istogramma dei valori con i conteggi, e
disegna le quadruple che realizzano una delta scelta.

Il grafo si cambia dallo yaml (`graph.type` + i suoi parametri), come in
experiments/synthetic/run_experiment.py: valgono anche cora/citeseer/pubmed
(un pezzo di n nodi del grafo reale) e mutag/proteins/... (il grafo
graph_index del benchmark TU, preso intero).

    python experiments/scratch/delta_histogram.py
"""

import itertools
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets')))

from src.graphs.utils import create_graph
from src.graphs.visualization import draw_graphs, draw_layout
from src.optimization.objectives import compute_distance_nodes, gromov_energy
from src.utils.config import REPO_ROOT, load_graph_config

from common import TU_DATASETS, citation_patch, tu_graph

# ----------------------------- config ---------------------------------
TARGET_DELTA = None                  # None = la delta massima; oppure es. 1.0
MAX_TO_DRAW = 9                      # quante quadruple disegnare al massimo
LOG_SCALE = False                    # asse y dell'istogramma in scala log
SAVE_DIR = None                      # es. 'data/scratch' (relativo alla repo)
# ----------------------------------------------------------------------

graph_cfg = load_graph_config()

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

# erdos_renyi/geometric possono venire disconnessi: le distanze sarebbero
# infinite e la delta non avrebbe senso, quindi si tiene solo la LCC
if not nx.is_connected(G):
    lcc = max(nx.connected_components(G), key=len)
    print(f"grafo disconnesso: tengo la LCC ({len(lcc)}/{G.number_of_nodes()} nodi)")
    G = G.subgraph(lcc).copy()

G.add_edge(4,5)
G.add_edge(1,4)
nodes = list(G.nodes())
dist_matrix, index = compute_distance_nodes(G)

# le quadruple sono in "index space": posizioni in list(G.nodes())
quads = list(itertools.combinations(range(len(nodes)), 4))
deltas = gromov_energy(quads, dist_matrix)

print(f"Grafo: {graph_cfg['type']}   n={len(nodes)}   |E|={G.number_of_edges()}   "
      f"seed={graph_cfg.get('seed')}")
print(f"Quadruple totali: {len(quads)}")
print(f"delta max (iperbolicita' di Gromov) = {deltas.max()}")
print(f"delta media                         = {deltas.mean():.4f}")

values, counts = np.unique(deltas, return_counts=True)
print("\n delta    count")
for v, c in zip(values, counts):
    print(f"{v:6.2f}  {c:8d}")

if SAVE_DIR is not None:
    save_dir = os.path.join(REPO_ROOT, SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
else:
    save_dir = None

# --- istogramma: una barra per ogni valore distinto di delta ----------
plt.figure(figsize=(8, 5))
bars = plt.bar([f"{v:g}" for v in values], counts,
               color='skyblue', edgecolor='black')
plt.bar_label(bars, fontsize=9, padding=2)
if LOG_SCALE:
    plt.yscale('log')
plt.title(f"Distribuzione di $\\delta$ su tutte le quadruple\n"
          f"{graph_cfg['type']} (n={len(nodes)}, |E|={G.number_of_edges()}), "
          f"{len(quads)} quadruple, $\\delta_{{max}}$={deltas.max():g}")
plt.xlabel("$\\delta$ della quadrupla")
plt.ylabel("numero di quadruple")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
if save_dir:
    plt.savefig(os.path.join(save_dir, f"hist_delta_{graph_cfg['type']}.png"), dpi=150)
plt.show()

# --- quadruple con la delta richiesta ---------------------------------
target = deltas.max() if TARGET_DELTA is None else TARGET_DELTA
selected = [q for q, d in zip(quads, deltas) if np.isclose(d, target)]

print(f"\nQuadruple con delta = {target:g}: {len(selected)}")
for q in selected[:20]:
    print("  ", tuple(nodes[i] for i in q))
if len(selected) > 20:
    print(f"   ... (altre {len(selected) - 20})")

if selected:
    to_draw = selected[:MAX_TO_DRAW]
    if len(selected) > MAX_TO_DRAW:
        print(f"Ne disegno solo {MAX_TO_DRAW}.")
    titles = [f"{tuple(nodes[i] for i in q)}  -  $\\delta$={target:g}" for q in to_draw]
    highlight_nodes = [[nodes[i] for i in q] for q in to_draw]
    fig, axes = draw_graphs([G] * len(to_draw), [pos] * len(to_draw),
                            titles=titles, highlight_nodes=highlight_nodes)
    if save_dir:
        fig.savefig(os.path.join(save_dir, f"quads_delta{target:g}_{graph_cfg['type']}.png"),
                    dpi=150, bbox_inches='tight')
