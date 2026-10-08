"""
Heatmap dei punteggi di iperbolicita' locale G*(v) scritti da
generate_features.py: una griglia di grafi disegnati, una riga per
grafo (o per nodo) e una colonna per geometric_temperature, con i nodi
colorati dal loro punteggio. Ogni pannello ha la sua scala di colori
(min/max di quel pannello), quindi si legge la *forma* spaziale del
punteggio, non la sua ampiezza.

Funziona su entrambe le famiglie di dataset:

  mutag / proteins / ...   una riga per grafo, scelti con GRAPHS.
  cora / citeseer / pubmed   il grafo e' uno solo ed enorme, quindi si
                             disegna una riga per nodo (NODES) mostrando
                             il suo ball K-hop, col nodo centrale
                             cerchiato di nero. K qui e' solo il raggio
                             *disegnato*: i punteggi restano quelli
                             calcolati sul grafo intero col k di
                             generate_features.

Per riusarlo su un altro file basta cambiare DATASET/CSV qui sotto (es.
mutag_node_metrics_beta01.csv); GRAPHS, NODES e TEMPS sono intervalli
(start, stop) alla python: es. (5, 9) per i grafi dal 6o al 9o.

    python experiments/datasets/plot_metric_heatmaps.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import (FEATURES_ROOT, TU_DATASETS, load_lcc, load_tu, to_nx,
                    _parse_bracket_array)

DATASET = "mutag"
CSV = os.path.join(FEATURES_ROOT, f"{DATASET}_node_metrics.csv")
META = CSV.replace(".csv", ".meta.json")
GRAPHS = (0, 3)   # dataset TU: quali grafi, come una slice: (5, 9) = il 6o..9o
NODES = (0, 2)    # cora/citeseer: quali nodi (id della LCC) fanno da centro
K = 8             # raggio del ball disegnato attorno a ogni nodo di NODES
TEMPS = (0, 5)    # quali geometric_temperature: (9, 15) = la 10a..15a

multi_graph = DATASET in TU_DATASETS
temp_ids = list(range(*TEMPS))

# i valori di geometric_temperature stanno nel .meta.json scritto accanto al csv
all_temps = json.load(open(META))["geometric_temperature"]

# ----------------------------------------------------------------------
# Una riga della griglia = (titolo, grafo da disegnare, i suoi nodi nello
# spazio di indici del csv, il blocco di profili di quei nodi, il centro)
# ----------------------------------------------------------------------
panels = []

if multi_graph:
    graphs = load_tu(DATASET)
    df = pd.read_csv(CSV).sort_values(["graph_id", "node_id"])

    for g in range(*GRAPHS):
        G = to_nx(graphs[g])
        rows = df[df["graph_id"] == g]
        profile = np.stack(rows["metric_result"].apply(_parse_bracket_array).values)
        panels.append((f"graph {g}", G, list(G.nodes()), profile, None))
else:
    data, _ = load_lcc(DATASET)
    G = to_nx(data)
    df = pd.read_csv(CSV).sort_values("node_id")
    profile = np.stack(df["metric_result"].apply(_parse_bracket_array).values)
    assert len(profile) == G.number_of_nodes(), \
        f"{CSV}: {len(profile)} righe ma la LCC ha {G.number_of_nodes()} nodi"

    for v in range(*NODES):
        # sottografo indotto sul ball K-hop di v (nodi a distanza <= K)
        ball = nx.ego_graph(G, v, radius=K)
        nodes = list(ball.nodes())
        # profile e' indicizzato per node_id, che qui e' l'id nella LCC
        panels.append((f"nodo {v} ({len(nodes)} nodi a <= {K} hop)",
                       ball, nodes, profile[nodes], v))

fig, axes = plt.subplots(len(panels), len(temp_ids),
                         figsize=(3.4 * len(temp_ids), 3.4 * len(panels)),
                         squeeze=False)
for row, (label, G, nodes, profile, center) in enumerate(panels):
    pos = nx.spring_layout(G, seed=42)
    # piu' il ball e' grande, piu' i pallini devono rimpicciolire per non
    # coprire tutto il pannello
    if len(nodes) < 60:
        size = 120
    elif len(nodes) < 300:
        size = 30
    else:
        size = 12

    for col, j in enumerate(temp_ids):
        ax = axes[row][col]
        scores = profile[:, j]
        # niente vmin/vmax: ogni pannello si normalizza sul proprio range
        drawn = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=scores,
                                       cmap=plt.cm.Reds, node_size=size, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7, ax=ax)
        if center is not None:
            nx.draw_networkx_nodes(G, pos, nodelist=[center], node_color="none",
                                   node_size=size * 4, edgecolors="black",
                                   linewidths=1.5, ax=ax)
        ax.set_title(f"{label}\n$T_{{geom}}$={all_temps[j]:.2f}", fontsize=9)
        ax.axis("off")
        fig.colorbar(drawn, ax=ax, shrink=0.7)

fig.suptitle(os.path.basename(CSV))
if multi_graph:
    save_path = CSV.replace(".csv", "_heatmap.png")
else:
    save_path = CSV.replace(".csv", f"_nodes{NODES[0]}-{NODES[1]}_k{K}_heatmap.png")
plt.savefig(save_path, bbox_inches="tight", dpi=150)
print(f"saved {save_path}")
plt.show()
