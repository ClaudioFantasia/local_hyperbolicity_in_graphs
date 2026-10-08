"""
Che raggio serve a un random geometric graph per avere un dato grado medio?

Il raggio da solo non basta a fissare la densita': nel quadrato unitario
il grado medio e' circa (n-1) * area del disco di raggio r, quindi dipende
anche da n. Fissato n, qui si cerca l'r che centra il grado medio voluto.

Due stime, entrambe stampate:
  - analitica: il valore medio dell'area disco-quadrato su centri uniformi
    vale  pi*r^2 - (8/3)*r^3 + r^4/2  (r <= 1), dove i termini negativi
    sono l'effetto bordo: i nodi vicini al bordo hanno meno vicini perche'
    parte del loro disco cade fuori dal quadrato. Da qui
        k(r) = (n-1) * (pi*r^2 - (8/3)*r^3 + r^4/2)
    e per una prima stima grezza si puo' invertire il solo termine pi*r^2:
        r ~ sqrt(k / ((n-1)*pi))
  - empirica: bisezione su r misurando il grado medio vero su SEEDS
    realizzazioni. E' questa la risposta da usare, l'analitica serve da
    controllo.

DEGREE_ON decide su cosa si misura il grado: 'lcc' (componente connessa
piu' grande, cioe' il grafo che poi si scora davvero) o 'full' (tutto il
grafo, isolati compresi). A raggi piccoli le due cose differiscono
parecchio, perche' gli isolati abbassano il grado medio di 'full'.

    python experiments/scratch/calibrate_geometric_radius.py
"""

import math
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_geometric_graph
from src.utils.config import REPO_ROOT

# ----------------------------- config ---------------------------------
N_VALUES = [50, 100, 200]           # per ogni n si calibra ogni target
TARGET_DEGREES = [6.0, 8.5, 12.0]   # gradi medi voluti
SEEDS = list(range(20))             # realizzazioni su cui si media il grado
DEGREE_ON = 'lcc'                   # 'lcc' o 'full'
TOL = 0.02                          # tolleranza sul grado medio
MAX_ITER = 40                       # iterazioni di bisezione
SAVE_DIR = None                     # es. 'data/scratch' (relativo alla repo)
# ----------------------------------------------------------------------


def analytic_degree(n, r):
    """Grado medio atteso nel quadrato unitario, con la correzione di bordo."""
    area = math.pi * r ** 2 - (8 / 3) * r ** 3 + r ** 4 / 2
    return (n - 1) * area


def measured_degree(n, r):
    """Grado medio misurato su SEEDS realizzazioni (piu' la sua std e n_LCC)."""
    degrees = []
    sizes = []
    for seed in SEEDS:
        G, _ = create_geometric_graph(n=n, radius=r, seed=seed)
        if DEGREE_ON == 'lcc':
            H = G.subgraph(max(nx.connected_components(G), key=len))
        else:
            H = G
        degrees.append(2 * H.number_of_edges() / H.number_of_nodes())
        sizes.append(H.number_of_nodes())
    return np.mean(degrees), np.std(degrees), np.mean(sizes)


def calibrate(n, target):
    """
    Bisezione su r: il grado medio cresce con r, quindi basta dimezzare
    l'intervallo finche' non si centra il target entro TOL.
    """
    lo, hi = 0.01, 0.7
    for _ in range(MAX_ITER):
        r = (lo + hi) / 2
        k, _, _ = measured_degree(n, r)
        if abs(k - target) < TOL:
            return r
        if k < target:
            lo = r
        else:
            hi = r
    return (lo + hi) / 2


print(f"grado medio misurato su: {DEGREE_ON}   ({len(SEEDS)} realizzazioni per punto)\n")
print(f"{'n':>5} {'k voluto':>9} {'r stima':>9} {'r calibrato':>12} "
      f"{'k ottenuto':>11} {'std k':>7} {'n_LCC':>7} {'k analitico':>12}")

calibrated = {}      # (n, target) -> raggio
for n in N_VALUES:
    for target in TARGET_DEGREES:
        r_guess = math.sqrt(target / ((n - 1) * math.pi))
        r = calibrate(n, target)
        k, k_std, n_lcc = measured_degree(n, r)
        calibrated[(n, target)] = r

        print(f"{n:>5} {target:>9.1f} {r_guess:>9.4f} {r:>12.4f} "
              f"{k:>11.2f} {k_std:>7.2f} {n_lcc:>7.1f} {analytic_degree(n, r):>12.2f}")

# righe pronte da incollare in delta_distribution_distance.py
print("\nMODELS pronti da copiare:")
for (n, target), r in calibrated.items():
    print(f"    {{'name': 'geom k={target:g}', 'type': 'geometric', "
          f"'n': {n}, 'geometric_radius': {r:.4f}}},")

# --- curva k(r): misurata (punti) contro analitica (linea) -------------
plt.figure(figsize=(8, 5.5))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
radii = np.linspace(0.05, 0.45, 15)

for i, n in enumerate(N_VALUES):
    measured = [measured_degree(n, r)[0] for r in radii]
    plt.plot(radii, measured, 'o-', ms=4, color=colors[i], label=f"n={n} (misurato)")
    plt.plot(radii, [analytic_degree(n, r) for r in radii], '--', lw=1,
             color=colors[i], label=f"n={n} (analitico)")

for target in TARGET_DEGREES:
    plt.axhline(target, color='gray', lw=0.8, ls=':')
for (n, target), r in calibrated.items():
    plt.plot(r, target, marker='*', ms=14, color='black', zorder=5)

plt.title(f"Grado medio ({DEGREE_ON}) in funzione del raggio\n"
          "stelle = raggi calibrati")
plt.xlabel("raggio")
plt.ylabel("grado medio")
plt.ylim(0, max(TARGET_DEGREES) * 3)
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()

if SAVE_DIR is not None:
    save_dir = os.path.join(REPO_ROOT, SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "calibrate_geometric_radius.png"),
                dpi=150, bbox_inches='tight')

plt.show()
