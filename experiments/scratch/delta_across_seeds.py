"""
Confronto della distribuzione di delta su piu' realizzazioni dello STESSO
random graph: parametri fissi (n, p / raggio), cambia solo il seed del
generatore. Serve a vedere quanto la delta di Gromov e' una proprieta'
stabile del modello e quanto invece e' rumore della singola realizzazione.

Nota sul seed: create_graph ha seed=42 di default e lo yaml fissa
graph.seed, quindi delta_histogram.py guarda sempre lo stesso identico
grafo. Qui il seed dello yaml viene ignorato di proposito: le
realizzazioni sono esattamente quelle elencate in SEEDS, cosi' l'esperimento
resta riproducibile ma non e' piu' un caso singolo.

I grafi random possono essere disconnessi (distanze infinite): si tiene
solo la componente connessa piu' grande, e la sua dimensione viene
stampata seed per seed (se varia molto, e' lei a spiegare le differenze).

    python experiments/scratch/delta_across_seeds.py
"""

import itertools
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_graph
from src.optimization.objectives import compute_distance_nodes, gromov_energy
from src.utils.config import REPO_ROOT, load_graph_config

# ----------------------------- config ---------------------------------
# i parametri di base vengono dallo yaml (sezione `graph`), OVERRIDES li
# sovrascrive: comodo per cambiare modello senza toccare il config file.
OVERRIDES = {'type': 'erdos_renyi', 'n': 100, 'p': 0.15}
OVERRIDES = {'type': 'geometric', 'n': 100, 'geometric_radius': 0.2}
# geometric: {'type': 'geometric', 'n': 50, 'geometric_radius': 0.25}

SEEDS = list(range(20))     # una realizzazione per seed
LOG_SCALE = False           # asse y in scala log nel pannello delle distribuzioni
SAVE_DIR = None             # es. 'data/scratch' (relativo alla repo)
# ----------------------------------------------------------------------

graph_cfg = load_graph_config()
graph_cfg.update(OVERRIDES)
graph_cfg.pop('seed', None)          # il seed lo passiamo noi, uno per realizzazione

print(f"modello: {graph_cfg['type']}   overrides: {OVERRIDES}")
print(f"{len(SEEDS)} realizzazioni, seeds {SEEDS[0]}..{SEEDS[-1]}\n")
print(f"{'seed':>5} {'n':>5} {'n_LCC':>6} {'|E|':>6} {'quads':>9} "
      f"{'delta max':>10} {'delta media':>12} {'frac delta=0':>13}")

all_values = set()          # unione dei valori distinti di delta su tutti i seed
results = []                # una riga per realizzazione

for seed in SEEDS:
    G, _ = create_graph(seed=seed, **graph_cfg)
    lcc = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc).copy()

    dist_matrix, _ = compute_distance_nodes(H)
    quads = list(itertools.combinations(range(H.number_of_nodes()), 4))
    deltas = gromov_energy(quads, dist_matrix)

    values, counts = np.unique(deltas, return_counts=True)
    all_values.update(values.tolist())
    frac_zero = counts[0] / len(quads) if np.isclose(values[0], 0.0) else 0.0

    results.append({'seed': seed,
                    'n_lcc': H.number_of_nodes(),
                    'n_edges': H.number_of_edges(),
                    'delta_max': deltas.max(),
                    'delta_mean': deltas.mean(),
                    'values': values,
                    'fracs': counts / len(quads)})

    print(f"{seed:>5} {G.number_of_nodes():>5} {H.number_of_nodes():>6} "
          f"{H.number_of_edges():>6} {len(quads):>9} {deltas.max():>10g} "
          f"{deltas.mean():>12.4f} {frac_zero:>13.4f}")

# --- riassunto sulle realizzazioni ------------------------------------
delta_max = np.array([r['delta_max'] for r in results])
delta_mean = np.array([r['delta_mean'] for r in results])

print(f"\ndelta max  : min={delta_max.min():g}  max={delta_max.max():g}  "
      f"media={delta_max.mean():.4f}  std={delta_max.std():.4f}")
print(f"delta media: min={delta_mean.min():.4f}  max={delta_mean.max():.4f}  "
      f"media={delta_mean.mean():.4f}  std={delta_mean.std():.4f}")

# --- distribuzioni allineate sulla stessa griglia di valori ------------
grid = np.array(sorted(all_values))
curves = []
for r in results:
    frac = np.zeros(len(grid))
    for v, f in zip(r['values'], r['fracs']):
        frac[np.searchsorted(grid, v)] = f
    curves.append(frac)
curves = np.array(curves)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

# 1) una curva per realizzazione + la media
ax = axes[0]
for frac, r in zip(curves, results):
    ax.plot(grid, frac, marker='o', ms=3, lw=1, color='steelblue', alpha=0.35)
ax.plot(grid, curves.mean(axis=0), marker='o', ms=5, lw=2, color='black',
        label='media sulle realizzazioni')
if LOG_SCALE:
    ax.set_yscale('log')
ax.set_title(f"Distribuzione di $\\delta$, {len(SEEDS)} realizzazioni")
ax.set_xlabel("$\\delta$ della quadrupla")
ax.set_ylabel("frazione di quadruple")
ax.grid(alpha=0.3)
ax.legend()

# 2) delta max realizzazione per realizzazione
ax = axes[1]
ax.bar([str(r['seed']) for r in results], delta_max,
       color='salmon', edgecolor='black')
ax.axhline(delta_max.mean(), color='black', ls='--', lw=1,
           label=f"media = {delta_max.mean():.2f}")
ax.set_title("$\\delta_{max}$ per seed")
ax.set_xlabel("seed")
ax.set_ylabel("$\\delta_{max}$")
ax.tick_params(axis='x', labelsize=8)
ax.grid(axis='y', alpha=0.3)
ax.legend()

# 3) quanto si sposta la delta media da una realizzazione all'altra
ax = axes[2]
ax.hist(delta_mean, bins=10, color='skyblue', edgecolor='black')
ax.axvline(delta_mean.mean(), color='black', ls='--', lw=1,
           label=f"media = {delta_mean.mean():.3f}\nstd = {delta_mean.std():.3f}")
ax.set_title("$\\delta$ media, distribuzione sulle realizzazioni")
ax.set_xlabel("$\\delta$ media della realizzazione")
ax.set_ylabel("numero di realizzazioni")
ax.grid(axis='y', alpha=0.3)
ax.legend()

fig.suptitle(f"{graph_cfg['type']}  " +
             "  ".join(f"{k}={v}" for k, v in OVERRIDES.items() if k != 'type'))
fig.tight_layout()

if SAVE_DIR is not None:
    save_dir = os.path.join(REPO_ROOT, SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"delta_across_seeds_{graph_cfg['type']}.png"),
                dpi=150, bbox_inches='tight')

plt.show()
