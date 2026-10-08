"""
Quanto sono diverse due distribuzioni di delta? Qui la domanda diventa
quantitativa: si generano piu' realizzazioni di piu' modelli di random
graph, si calcola la distribuzione di delta di ognuna e poi la distanza
fra TUTTE le coppie di realizzazioni.

L'idea e' che la distanza fra due realizzazioni dello stesso modello
(rumore del campionamento) faccia da unita' di misura per giudicare la
distanza fra modelli diversi: se W1(ER, geometric) e' 10 volte la W1
tipica fra due ER, i due modelli sono davvero distinguibili; se e' 1.5
volte, la differenza e' dentro il rumore.

Tre distanze, tutte stampate:
  - W1 (Wasserstein-1): quanto va spostata in media la massa per passare
    da una distribuzione all'altra, in unita' di delta. Tiene conto di
    QUANTO sono lontani i valori, non solo di quali sono diversi -> e' la
    piu' sensata qui, perche' delta e' ordinato.
  - W2 (Wasserstein-2): come W1 ma il costo dello spostamento e' al
    quadrato, quindi pesa di piu' le code (le quadruple con delta alta).
    W2 >= W1 sempre: se il rapporto W2/W1 e' grande, la differenza fra le
    due distribuzioni sta soprattutto nella coda.
  - TV (variazione totale): 0 = identiche, 1 = supporti disgiunti.
    Non guarda l'ordine dei valori, ma e' facile da leggere.

Per ogni coppia di modelli si stampa la distanza media fra i due e quella
interna a ciascuno dei due: la prima dice quanto sono lontani, la seconda
da' la scala per giudicare se quella distanza e' tanta o poca.

Attenzione al confronto: modelli con densita' diversa hanno diametro
diverso, quindi una parte della distanza e' solo "grafo piu' o meno
denso" e non struttura. La tabella stampa n_LCC e grado medio proprio
per tenere d'occhio questo confondimento (per un confronto pulito,
scegliere i parametri in modo che il grado medio sia simile).

    python experiments/scratch/delta_distribution_distance.py
"""

import itertools
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import ot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_graph
from src.optimization.objectives import compute_distance_nodes, gromov_energy
from src.utils.config import REPO_ROOT, load_graph_config

# ----------------------------- config ---------------------------------
# ogni voce e' un modello: 'name' e' solo l'etichetta, il resto sovrascrive
# i parametri dello yaml. Il seed NON si mette qui: lo fa variare SEEDS.
#
# I parametri sono scelti in modo che i modelli si confrontino a PARITA' di
# grado medio k della LCC (k ~ (n-1)*p per ER, misurato per il geometric):
# altrimenti buona parte della distanza sarebbe solo differenza di densita'.
# Le due coppie (k~8.5 e k~12.2) danno anche un controllo: quanto pesa la
# densita' rispetto al modello.
MODELS = [
    {'name': 'ER k=2.5',    'type': 'erdos_renyi', 'n': 50, 'p': 0.05},
    {'name': 'ER k=5',  'type': 'erdos_renyi',   'n': 100, 'p': 0.05},
    {'name': 'ER k=7.5',   'type': 'erdos_renyi', 'n': 150, 'p': 0.05},
    #{'name': 'ER k=10', 'type': 'erdos_renyi',   'n': 200, 'geometric_radius': 0.05},
]
SEEDS = list(range(10))     # stessi seed per tutti i modelli
METRIC = 'W1'               # 'W1', 'W2' o 'TV': quale usare nella heatmap
SAVE_DIR = None           # es. 'data/scratch' (relativo alla repo)
# ----------------------------------------------------------------------


def wasserstein(p, q, grid, order):
    """
    Distanza di Wasserstein di ordine 1 o 2 fra due distribuzioni discrete
    sulla stessa griglia di valori di delta (POT, caso 1-D).

    order=1: il costo di spostare la massa e' la distanza percorsa; si
             legge come "di quanto va spostata in media la massa", in
             unita' di delta.
    order=2: il costo e' la distanza al quadrato, quindi pesa di piu' gli
             spostamenti lunghi, cioe' le code (le quadruple con delta
             alta). Vale sempre W2 >= W1.

    NB: ot.wasserstein_1d restituisce il COSTO W_order^order, non la
    distanza -> per la W2 va presa la radice.
    """
    cost = ot.wasserstein_1d(grid, grid, p, q, p=order)
    return float(cost) ** (1 / order)


def total_variation(p, q):
    """Distanza in variazione totale: meta' della L1 fra le due densita'."""
    dist = 0.0
    for i in range(len(p)):
        dist += abs(p[i] - q[i])
    return dist / 2.0


# --------------------- distribuzioni di delta -------------------------
base_cfg = load_graph_config()

labels = []        # (nome modello, seed) di ogni realizzazione
model_of = []      # indice del modello a cui appartiene la realizzazione
values_per_run = []
counts_per_run = []
stats = []         # n_lcc, grado medio, delta max, delta media

print(f"{'modello':<14} {'seed':>5} {'n_LCC':>6} {'grado medio':>12} "
      f"{'delta max':>10} {'delta media':>12}")

for m_idx, model in enumerate(MODELS):
    cfg = dict(base_cfg)
    cfg.update(model)
    name = cfg.pop('name')
    cfg.pop('seed', None)          # il seed lo passiamo noi

    for seed in SEEDS:
        G, _ = create_graph(seed=seed, **cfg)
        lcc = max(nx.connected_components(G), key=len)
        H = G.subgraph(lcc).copy()

        dist_matrix, _ = compute_distance_nodes(H)
        quads = list(itertools.combinations(range(H.number_of_nodes()), 4))
        deltas = gromov_energy(quads, dist_matrix)

        values, counts = np.unique(deltas, return_counts=True)
        labels.append((name, seed))
        model_of.append(m_idx)
        values_per_run.append(values)
        counts_per_run.append(counts / len(quads))

        avg_degree = 2 * H.number_of_edges() / H.number_of_nodes()
        stats.append((H.number_of_nodes(), avg_degree, deltas.max(), deltas.mean()))
        print(f"{name:<14} {seed:>5} {H.number_of_nodes():>6} {avg_degree:>12.2f} "
              f"{deltas.max():>10g} {deltas.mean():>12.4f}")

# tutte le distribuzioni sulla stessa griglia di valori di delta
grid = np.array(sorted(set(v for values in values_per_run for v in values)))
distributions = []
for values, fracs in zip(values_per_run, counts_per_run):
    p = np.zeros(len(grid))
    for v, f in zip(values, fracs):
        p[np.searchsorted(grid, v)] = f
    distributions.append(p)

# ------------------- matrice delle distanze ---------------------------
n_runs = len(distributions)
matrices = {'W1': np.zeros((n_runs, n_runs)),
            'W2': np.zeros((n_runs, n_runs)),
            'TV': np.zeros((n_runs, n_runs))}
for i in range(n_runs):
    for j in range(i + 1, n_runs):
        p, q = distributions[i], distributions[j]
        matrices['W1'][i, j] = matrices['W1'][j, i] = wasserstein(p, q, grid, 1)
        matrices['W2'][i, j] = matrices['W2'][j, i] = wasserstein(p, q, grid, 2)
        matrices['TV'][i, j] = matrices['TV'][j, i] = total_variation(p, q)

D = matrices[METRIC]


def mean_within(M, idx):
    """Distanza media fra due realizzazioni dello stesso gruppo (coppie i<j)."""
    vals = []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            vals.append(M[idx[i], idx[j]])
    return float(np.mean(vals))


def mean_between(M, idx_a, idx_b):
    """Distanza media fra una realizzazione di un gruppo e una dell'altro."""
    vals = []
    for i in idx_a:
        for j in idx_b:
            vals.append(M[i, j])
    return float(np.mean(vals))


runs_of_model = [[i for i in range(n_runs) if model_of[i] == m]
                 for m in range(len(MODELS))]

print(f"\nDistanza media fra distribuzioni ({METRIC}). "
      f"Diagonale = fra realizzazioni dello stesso modello (= rumore).")
print(f"{'':<14}" + "".join(f"{m['name']:>14}" for m in MODELS))
for a, model_a in enumerate(MODELS):
    row = f"{model_a['name']:<14}"
    for b in range(len(MODELS)):
        if a == b:
            row += f"{mean_within(D, runs_of_model[a]):>14.4f}"
        else:
            row += f"{mean_between(D, runs_of_model[a], runs_of_model[b]):>14.4f}"
    print(row)

# --------------------- riepilogo per coppie ---------------------------
# Il rumore interno e' una proprieta' del singolo modello, quindi sta in
# una tabella sua invece di essere ripetuto su ogni coppia.
print("\nRumore interno: distanza media fra due realizzazioni dello stesso modello")
print(f"{'modello':<14} {'W1':>9} {'W2':>9} {'TV':>9} {'W2/W1':>8}")
for a, model_a in enumerate(MODELS):
    w1 = mean_within(matrices['W1'], runs_of_model[a])
    w2 = mean_within(matrices['W2'], runs_of_model[a])
    tv = mean_within(matrices['TV'], runs_of_model[a])
    print(f"{model_a['name']:<14} {w1:>9.4f} {w2:>9.4f} {tv:>9.4f} {w2 / w1:>8.2f}")

"""
# Le distanze fra modelli sono confrontabili fra righe; il rumore interno
# qui sopra da' la scala per giudicare se sono tante o poche. W2/W1 alto
# vuol dire che la differenza sta soprattutto nelle code.
print("\nDistanza fra modelli (da confrontare con il rumore interno qui sopra)")
print(f"{'coppia di modelli':<30} {'W1':>9} {'W2':>9} {'TV':>9} {'W2/W1':>8}")
for a in range(len(MODELS)):
    for b in range(a + 1, len(MODELS)):
        w1 = mean_between(matrices['W1'], runs_of_model[a], runs_of_model[b])
        w2 = mean_between(matrices['W2'], runs_of_model[a], runs_of_model[b])
        tv = mean_between(matrices['TV'], runs_of_model[a], runs_of_model[b])

        pair = f"{MODELS[a]['name']} vs {MODELS[b]['name']}"
        print(f"{pair:<30} {w1:>9.4f} {w2:>9.4f} {tv:>9.4f} {w2 / w1:>8.2f}")
"""
# ----------------------------- figure ---------------------------------
colors = plt.cm.tab10(np.linspace(0, 1, 10))
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 1) le distribuzioni, un colore per modello
ax = axes[0]
for i, p in enumerate(distributions):
    m_idx = model_of[i]
    ax.plot(grid, p, marker='o', ms=3, lw=1, alpha=0.6, color=colors[m_idx],
            label=MODELS[m_idx]['name'] if i == runs_of_model[m_idx][0] else None)
ax.set_title(f"$\\delta$ distribution, {len(SEEDS)} realization per model")
ax.set_xlabel("$\\delta$ of the 4-tuple")
ax.set_ylabel("4-tuple fraction")
ax.grid(alpha=0.3)
ax.legend()

# 2) matrice delle distanze fra tutte le realizzazioni: i blocchi sulla
#    diagonale (stesso modello) devono essere piu' chiari del resto
ax = axes[1]
im = ax.imshow(D, cmap='viridis')
fig.colorbar(im, ax=ax, label=f"distance {METRIC}")
for m in range(1, len(MODELS)):
    ax.axhline(m * len(SEEDS) - 0.5, color='white', lw=1.5)
    ax.axvline(m * len(SEEDS) - 0.5, color='white', lw=1.5)
ticks = [m * len(SEEDS) + len(SEEDS) / 2 - 0.5 for m in range(len(MODELS))]
ax.set_xticks(ticks)
ax.set_xticklabels([m['name'] for m in MODELS], rotation=20)
ax.set_yticks(ticks)
ax.set_yticklabels([m['name'] for m in MODELS])
ax.set_title(f"Distance {METRIC} across all the realization\n"
             "(diagonal = same model)")

fig.tight_layout()

if SAVE_DIR is not None:
    save_dir = os.path.join(REPO_ROOT, SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"delta_distribution_distance_{METRIC}.png"),
                dpi=150, bbox_inches='tight')

plt.show()
