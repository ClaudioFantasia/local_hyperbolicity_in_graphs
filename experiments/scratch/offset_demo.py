"""
I due casi limite che rendono visibile l'offset geometrico dell'entropic
score (vedi offset_entropica_vs_KL.md, sezioni 5 e 6):

  - path: tutte le delta sono 0, quindi KL = 0 esatto ovunque mentre
    l'entropico e' *identicamente* il suo offset B(v) = beta*log Z_gamma(v),
    cioe' una misura di centralita';
  - lattice: con il grafo intero come vicinato la KL dice angolo > centro,
    l'entropico dice il contrario perche' l'offset lo ribalta; con k=4
    entra in piu' la troncatura (la palla dell'angolo non contiene le
    quadruple ad alta delta).

Stampa nodo per nodo KL, entropico, offset e la loro differenza.

    python experiments/scratch/offset_demo.py
"""

import os
import sys

from scipy.special import logsumexp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_graph
from src.optimization.local import KL_score, entropic_score
from src.optimization.neighborhood import get_neighborhood, sampling_quads
from src.optimization.objectives import (compute_distance_nodes, gromov_energy,
                                         tuple_distance)

# ----------------------------- config ---------------------------------
BETA = 1.0        # temperatura del rilassamento
LAMBDA = 1.0      # peso di d_v nel costo entropico (= beta / T_geom)
# ----------------------------------------------------------------------


def table(G, k, label):
    dist_matrix, index = compute_distance_nodes(G)
    print(f"\n--- {label} (k={k}, beta={BETA}, lambda={LAMBDA}) ---")
    print(f"{'v':>3} {'|H|':>7} {'max d':>6} {'KL':>8} {'entropico':>10} "
          f"{'offset':>8} {'entr-off':>9}")

    for v in G.nodes():
        neighborhood = get_neighborhood(G, index[v], k)
        quads = sampling_quads(neighborhood)
        w = tuple_distance(quads, dist_matrix, index[v])
        deltas = gromov_energy(quads, dist_matrix)

        # B(v): la parte di V* che non contiene nessuna delta
        offset = BETA * logsumexp(-LAMBDA * w / BETA)
        kl = KL_score(G, index[v], {}, k, BETA, BETA / LAMBDA, dist_matrix)[0]
        entropic = entropic_score(G, index[v], {}, k, BETA, LAMBDA, dist_matrix)

        print(f"{v:>3} {len(quads):>7} {deltas.max():>6g} {kl:>8.4f} "
              f"{entropic:>10.4f} {offset:>8.4f} {entropic - offset:>9.4f}")


G, _ = create_graph(type='path', n=15)
table(G, 4, "path n=15, palla 4-hop")
table(G, 50, "path n=15, grafo intero")

G, _ = create_graph(type='lattice', n=5, m=5)
table(G, 4, "lattice 5x5, palla 4-hop")
table(G, 50, "lattice 5x5, grafo intero")
