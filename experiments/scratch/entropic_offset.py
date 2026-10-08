"""
Quanto dell'entropic score e' iperbolicita' e quanto e' solo geometria?

    V*(v) = T * logsumexp_i[(delta_i - lambda*d_v(h_i)) / T]
          = V*_KL(v; T_geom = T/lambda)  +  T * log Z_v
    Z_v   = sum_i exp(-lambda * d_v(h_i) / T)

Il secondo termine non contiene nessuna delta: e' un puro termine di
volume/distanza della palla attorno a v. Questo script, per una griglia
di lambda, stampa V*, l'offset T*log Z_v e il segnale V* - offset nodo
per nodo, e misura quanta della varianza di V* fra i nodi e' offset.

    python experiments/scratch/entropic_offset.py
"""

import os
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import spearmanr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets')))

from src.graphs.utils import create_graph
from src.graphs.visualization import draw_layout
from src.optimization.local import KL_score, entropic_score
from src.optimization.neighborhood import get_neighborhood, sampling_quads
from src.optimization.objectives import (compute_distance_nodes, gromov_energy,
                                         tuple_distance)
from src.utils.config import load_config

from common import TU_DATASETS, citation_patch, tu_graph

# ----------------------------- config ---------------------------------
LAMBDAS = [0, 0.5, 1, 2, 5]   # peso di d_v nel costo
K = 4                         # raggio del vicinato (indipendente dallo yaml)
# ----------------------------------------------------------------------

cfg = load_config()
graph_cfg = cfg["graph"]
T = cfg["optimization"]["temperature"]

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

print(f"Grafo: {graph_cfg['type']}   n={len(nodes)}   |E|={G.number_of_edges()}   "
      f"k={K}   T={T}")

for lam in LAMBDAS:
    scores, offsets, sizes = [], [], []
    for v in nodes:
        neighborhood = get_neighborhood(G, index[v], K)
        quads = sampling_quads(neighborhood)
        w = tuple_distance(quads, dist_matrix, index[v])

        # T * log Z_v: la parte di V* che non contiene nessuna delta
        offsets.append(T * logsumexp(-lam * w / T))
        scores.append(entropic_score(G, index[v], {}, K, T, lam, dist_matrix))
        sizes.append(len(quads))

    scores = np.array(scores)
    offsets = np.array(offsets)
    signal = scores - offsets

    # il segnale deve coincidere con la KL a T_geom = T/lambda (lambda > 0)
    if lam > 0:
        kl = np.array([KL_score(G, index[v], {}, K, T, T / lam, dist_matrix)[0]
                       for v in nodes])
        check = f"max|segnale - KL(T_geom=T/lambda)| = {np.abs(signal - kl).max():.2e}"
    else:
        check = "lambda=0: nessun termine di distanza, offset = T*log|H|"

    print(f"\n{'='*70}\nlambda = {lam}\n{'='*70}")
    print(f"V*      : range [{scores.min():7.3f}, {scores.max():7.3f}]  std = {scores.std():.4f}")
    print(f"offset  : range [{offsets.min():7.3f}, {offsets.max():7.3f}]  std = {offsets.std():.4f}")
    print(f"segnale : range [{signal.min():7.3f}, {signal.max():7.3f}]  std = {signal.std():.4f}")
    print(f"corr(V*, offset)  = {np.corrcoef(scores, offsets)[0, 1]:+.4f}")
    print(f"corr(V*, segnale) = {np.corrcoef(scores, signal)[0, 1]:+.4f}")
    print(f"corr(V*, |H|)     = {np.corrcoef(scores, sizes)[0, 1]:+.4f}")
    print(f"Spearman(V*, segnale) = {spearmanr(scores, signal).statistic:+.4f}  "
          f"(quanto il ranking dei nodi sopravvive all'offset)")
    print(check)
