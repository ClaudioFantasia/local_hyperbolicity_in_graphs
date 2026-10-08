"""
Distanza fra due grafi prototipo: Gromov-Hausdorff e Gromov-Wasserstein.

I due grafi si vedono come spazi metrici finiti (nodi + distanza di
cammino minimo) e si confrontano in due modi:

  Gromov-Hausdorff   d_GH = 1/2 * min_R dis(R) sulle corrispondenze R
                     fra i due insiemi di nodi. Calcolata esatta per
                     forza bruta usando 2*d_GH = min_{f,g} max(dis f,
                     dis g, codis(f,g)) con f: A->B, g: B->A: costa
                     |B|^|A| + |A|^|B| mappe, quindi va bene fino a ~6
                     nodi per lato, non oltre.

  Gromov-Wasserstein rilassamento "a trasporto": invece di una
                     corrispondenza secca si cerca un accoppiamento T
                     fra le distribuzioni uniformi sui nodi. Usa POT
                     (`pip install pot`), ot.gromov.gromov_wasserstein.

Alla fine, per ciascuno dei due grafi, la stessa cosa che fa
experiments/scratch/delta_histogram.py: delta di Gromov esatta su tutte
le C(n,4) quadruple, con istogramma dei valori e conteggi.

I due grafi si scelgono nel blocco di config qui sotto, con le stesse
chiavi di configs/optimization_parameters.yaml (sezione `graph`).

    python experiments/scratch/gh_gw_prototypes.py
"""

import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import ot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_graph
from src.graphs.visualization import draw_graphs
from src.optimization.objectives import compute_distance_nodes, gromov_energy
from src.utils.config import REPO_ROOT

# ----------------------------- config ---------------------------------
GRAPH_A = {"type": "complete", "n": 5}
GRAPH_B = {"type": "complete",  "n": 8}
LOG_SCALE = False                    # asse y degli istogrammi in scala log
SAVE_DIR = None                      # es. 'data/scratch' (relativo alla repo)
# ----------------------------------------------------------------------

GA, posA = create_graph(**GRAPH_A)
GB, posB = create_graph(**GRAPH_B)
#GB.add_edge(4,5)

dA, _ = compute_distance_nodes(GA)
dB, _ = compute_distance_nodes(GB)
nA, nB = len(GA), len(GB)

print(f"A: {GRAPH_A['type']}  n={nA}  |E|={GA.number_of_edges()}")
print(f"B: {GRAPH_B['type']}  n={nB}  |E|={GB.number_of_edges()}")

# --------------------------- Gromov-Hausdorff -------------------------
# dis(f) = max_{x,x'} | d_A(x,x') - d_B(f(x),f(x')) |
def distortion(f, d_from, d_to):
    worst = 0.0
    for x in range(len(f)):
        for y in range(len(f)):
            worst = max(worst, abs(d_from[x, y] - d_to[f[x], f[y]]))
    return worst


# codis(f,g) = max_{x,y} | d_A(x,g(y)) - d_B(f(x),y) |: misura quanto f e g
# sono "l'una l'inversa dell'altra" in senso metrico
def codistortion(f, g, dA, dB):
    worst = 0.0
    for x in range(len(f)):
        for y in range(len(g)):
            worst = max(worst, abs(dA[x, g[y]] - dB[f[x], y]))
    return worst


# tutte le mappe A->B e B->A, ordinate per distorsione crescente: cosi'
# il minimo si trova subito e i due break tagliano quasi tutte le coppie
maps_AB = [(f, distortion(f, dA, dB)) for f in itertools.product(range(nB), repeat=nA)]
maps_BA = [(g, distortion(g, dB, dA)) for g in itertools.product(range(nA), repeat=nB)]
maps_AB.sort(key=lambda t: t[1])
maps_BA.sort(key=lambda t: t[1])

best = np.inf
best_pair = None
for f, dis_f in maps_AB:
    if dis_f >= best:
        break
    for g, dis_g in maps_BA:
        if max(dis_f, dis_g) >= best:
            break
        dis_R = max(dis_f, dis_g, codistortion(f, g, dA, dB))
        if dis_R < best:
            best = dis_R
            best_pair = (f, g)

gh = best / 2
print(f"\nGromov-Hausdorff: d_GH = {gh:g}   (distorsione minima = {best:g})")
f, g = best_pair
print("  mappa A->B:", {a: f[a] for a in range(nA)})
print("  mappa B->A:", {b: g[b] for b in range(nB)})

# -------------------------- Gromov-Wasserstein ------------------------
# distribuzioni uniformi sui nodi dei due grafi
pA = np.ones(nA) / nA
pB = np.ones(nB) / nB

T, log = ot.gromov.gromov_wasserstein(dA, dB, pA, pB, 'square_loss', log=True)
gw_loss = log['gw_dist']            # sum_{ijkl} |d_A(i,k)-d_B(j,l)|^2 T_ij T_kl

print(f"\nGromov-Wasserstein: loss = {gw_loss:.6f}   GW_2 = {0.5 * np.sqrt(gw_loss):.6f}")
print("accoppiamento T (righe = nodi di A, colonne = nodi di B):")
print(np.round(T * nA, 3))          # *n per leggerlo come una quasi-permutazione

# ------------------- delta di Gromov su tutte le quadruple ------------
if SAVE_DIR is not None:
    save_dir = os.path.join(REPO_ROOT, SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
else:
    save_dir = None

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, G, cfg, dist_matrix in [(axes[0], GA, GRAPH_A, dA), (axes[1], GB, GRAPH_B, dB)]:
    nodes = list(G.nodes())
    quads = list(itertools.combinations(range(len(nodes)), 4))
    deltas = gromov_energy(quads, dist_matrix)

    print(f"\n--- {cfg['type']} (n={len(nodes)}, |E|={G.number_of_edges()}) ---")
    print(f"Quadruple totali: {len(quads)}")
    print(f"delta max (iperbolicita' di Gromov) = {deltas.max()}")
    print(f"delta media                         = {deltas.mean():.4f}")

    values, counts = np.unique(deltas, return_counts=True)
    print("\n delta    count")
    for v, c in zip(values, counts):
        print(f"{v:6.2f}  {c:8d}")

    bars = ax.bar([f"{v:g}" for v in values], counts, color='skyblue', edgecolor='black')
    ax.bar_label(bars, fontsize=9, padding=2)
    if LOG_SCALE:
        ax.set_yscale('log')
    ax.set_title(f"{cfg['type']} (n={len(nodes)}, |E|={G.number_of_edges()})\n"
                 f"{len(quads)} quadruple, $\\delta_{{max}}$={deltas.max():g}")
    ax.set_xlabel("$\\delta$ della quadrupla")
    ax.set_ylabel("numero di quadruple")
    ax.grid(axis='y', alpha=0.3)

fig.suptitle(f"$d_{{GH}}$ = {gh:g}     $GW_2$ = {0.5 * np.sqrt(gw_loss):.4f}")
plt.tight_layout()
if save_dir:
    fig.savefig(os.path.join(save_dir, f"hist_delta_{GRAPH_A['type']}_vs_{GRAPH_B['type']}.png"), dpi=150)
plt.show()

draw_graphs([GA, GB], [posA, posB],
            titles=[f"A: {GRAPH_A['type']}", f"B: {GRAPH_B['type']}"])
