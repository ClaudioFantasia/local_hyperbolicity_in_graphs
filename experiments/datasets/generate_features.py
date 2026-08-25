"""
Compute the per-node local hyperbolicity profile (KL_score swept over a
range of geometric temperatures) for a dataset and write it to
data/hyperbolic_features/<dataset>_node_metrics.csv.

Two families of dataset are handled by the same command:

  cora / citeseer / pubmed   one big graph. Its largest connected
                             component is scored and the csv has two
                             columns, `node_id, metric_result`.
  mutag / proteins / ...     a collection of small graphs. Every node of
                             every graph is scored and the csv gains a
                             leading `graph_id` column.

Results are appended row by row and the script resumes from whatever is
already in the csv, so it can be stopped and restarted.

    python experiments/datasets/generate_features.py --dataset cora
    python experiments/datasets/generate_features.py --dataset mutag

Con --strategy increasing_neighborhood il ball k-hop viene campionato
(al piu' --m vicini per nodo per hop) invece che preso tutto: su Cora
con k=4 si passa da ~40s a ~0.3s per nodo. L'output finisce in un csv
separato (<dataset>_node_metrics_m<m>.csv) per non sovrascrivere quello
gia' calcolato col ball completo. Sui grafi TU non serve: sono cosi'
piccoli che il ball completo e' sempre esatto e istantaneo.

--k sovrascrive il k di SCORE_PARAMS e scrive in un csv suffissato _k<k>,
cosi' si possono tenere piu' raggi affiancati senza sovrascriverli.
"""

import argparse
import csv
import json
import os
import sys

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.optimization.local import KL_score

from common import DATASETS, TU_DATASETS, load_lcc, load_tu, metrics_path, to_nx

# il profilo finisce nel csv come str(array), e str() tronca con '...' sopra i
# 1000 elementi: con le griglie T_geom di oggi (25-50 valori) non succede, ma
# se un giorno le raffini oltre i 1000 il csv diventa illeggibile a calcolo
# gia' fatto
np.set_printoptions(threshold=np.inf)

# KL_score hyperparameters, tuned per dataset. geometric_temperature is
# swept as an array: KL_score returns one value per T_geom, so each node's
# row in the csv is the whole multi-scale profile.
SCORE_PARAMS = {
    "cora": {
        "k": 5,
        "temperature": 0.1,
        "geometric_temperature": np.arange(0.1, 2.6, 0.1),
    },
    "citeseer": {
        "k": 8,
        "temperature": 0.1,
        "geometric_temperature": np.arange(0.05, 2.51, 0.05),
    },
    # NCI1 cannot afford the whole-graph ball that TU_DEFAULT_PARAMS below
    # uses. Its graphs run to 111 nodes, so scoring every node against every
    # quad of its own graph is 2.5e10 quad-node pairs (~37h at the rate MUTAG
    # runs at) and peaks at 1.6 GB for a single node's logsumexp -- and on the
    # 7 largest graphs sampling_quads would blow past its own cap anyway, so
    # the score would stop being exact exactly where it costs most.
    #
    # So here k is a real hyperparameter again. k=5 is the structural analogue
    # of MUTAG's k=3: the ball is a median 15 nodes out of a median 27, i.e.
    # 55% of the graph, against MUTAG's 54%.
    "nci1": {
        "k": 5,
        "temperature": 0.1,
        "geometric_temperature": np.arange(0.05, 2.51, 0.05),
    },
}

# Default for any TU benchmark without an entry above. k is deliberately
# larger than any of these graphs (MUTAG has diameter <= 15), so the k-hop
# ball is the whole graph and k stops being a hyperparameter at all: every
# node sees every quad, and what makes G*(v) a *local* quantity is only the
# geometric weighting exp(-d_v(h)/T_geom), which still depends on v.
#
# Two things follow. The score is exact -- comb(28, 4) = 20475 quads is
# under sampling_quads' cap, so nothing is sampled. And T_geom becomes the
# single knob that sets the scale the score looks at, which is why it is
# swept finely here: the 50 values are a multi-scale positional encoding of
# the node, from nearly-local (0.05) to nearly-global (2.50).
TU_DEFAULT_PARAMS = {
    "k": 50,
    "temperature": 0.1,
    "geometric_temperature": np.arange(0.05, 2.51, 0.05),
}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=sorted(DATASETS) + sorted(TU_DATASETS),
                    default="cora")
parser.add_argument("--k", type=int, default=None,
                    help="Override the neighborhood radius of SCORE_PARAMS.")
parser.add_argument("--strategy", choices=["full_neighborhood", "increasing_neighborhood"],
                    default="full_neighborhood")
parser.add_argument("--m", type=int, default=3,
                    help="Vicini tenuti per nodo per hop (solo con "
                         "--strategy increasing_neighborhood).")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed del campionamento, cosi' la corsa e' riproducibile.")
args = parser.parse_args()

multi_graph = args.dataset in TU_DATASETS

if multi_graph:
    params = dict(SCORE_PARAMS.get(args.dataset, TU_DEFAULT_PARAMS))
else:
    params = dict(SCORE_PARAMS[args.dataset])
if args.k is not None:
    params["k"] = args.k

output_file = metrics_path(args.dataset)
if args.strategy == "increasing_neighborhood":
    output_file = output_file.replace(".csv", f"_m{args.m}.csv")
if args.k is not None:
    output_file = output_file.replace(".csv", f"_k{args.k}.csv")

# ----------------------------------------------------------------------
# The graphs to score, as a list either way: one LCC, or every TU graph
# ----------------------------------------------------------------------
if multi_graph:
    graphs = [to_nx(d) for d in load_tu(args.dataset)]
    columns = ["graph_id", "node_id", "metric_result"]
else:
    data, node_map = load_lcc(args.dataset)
    G = to_nx(data)
    assert nx.number_connected_components(G) == 1, "LCC extraction failed"
    graphs = [G]
    columns = ["node_id", "metric_result"]

total_nodes = sum(G.number_of_nodes() for G in graphs)

# Impronta di cosa c'e' dentro il csv: il grafo su cui e' stato calcolato e i
# parametri che danno significato alle colonne. Il nome del file codifica solo
# m e k, e solo se passati da CLI; senza questo file, per sapere cosa contiene
# un csv di tre mesi fa bisognerebbe leggere SCORE_PARAMS nella versione di
# allora. common.check_meta lo rilegge a ogni run che usa queste feature.
meta_file = output_file.replace(".csv", ".meta.json")
meta = {
    "dataset": args.dataset,
    "num_nodes": total_nodes,
    "num_edges": sum(G.number_of_edges() for G in graphs),
    "num_graphs": len(graphs),
    "k": params["k"],
    "temperature": params["temperature"],
    "geometric_temperature": params["geometric_temperature"].tolist(),
    "strategy": args.strategy,
    # m conta solo quando il ball e' campionato: registrarlo sempre farebbe
    # scattare il controllo qui sotto anche per un --m cambiato a vuoto
    "m": args.m if args.strategy == "increasing_neighborhood" else None,
    "seed": args.seed,
}

# resume: skip whatever is already in the csv
completed = set()
if os.path.exists(output_file):
    with open(output_file) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                completed.add((int(row[0]), int(row[1])) if multi_graph
                              else (0, int(row[0])))
    print(f"Resuming: {len(completed)} of {total_nodes} nodes already done")
else:
    with open(output_file, 'w', newline='') as f:
        csv.writer(f).writerow(columns)

# un resume ha senso solo se riprende lo stesso calcolo: se i parametri sono
# cambiati le righe vecchie e quelle nuove non sono la stessa quantita', e il
# csv misto non lo segnalerebbe nessuno
if os.path.exists(meta_file):
    with open(meta_file) as f:
        previous = json.load(f)
    if previous != meta:
        changed = [key for key in meta if previous.get(key) != meta[key]]
        raise SystemExit(
            f"{os.path.basename(meta_file)} dice che {os.path.basename(output_file)} "
            f"e' stato generato con parametri diversi ({', '.join(changed)}). "
            f"Cancella i due file per ricalcolarlo da zero, oppure usa --k / "
            f"--strategy per scrivere in un csv suffissato diverso."
        )

with open(meta_file, "w") as f:
    json.dump(meta, f, indent=2)

strategy_note = (f"{args.strategy} (m={args.m}, seed={args.seed})"
                 if args.strategy == "increasing_neighborhood" else args.strategy)
print(f"k={params['k']}  temperature={params['temperature']}  "
      f"T_geom: {len(params['geometric_temperature'])} values in "
      f"[{params['geometric_temperature'][0]:.2f}, "
      f"{params['geometric_temperature'][-1]:.2f}]  {strategy_note}")

progress = tqdm(total=total_nodes)
for graph_id, G in enumerate(graphs):
    # scipy's shortest_path on the adjacency matrix is much faster here than
    # the networkx BFS in src.optimization.objectives; the row/column order is
    # G.nodes() == 0..n-1, i.e. the index space KL_score expects
    dist_matrix = shortest_path(nx.adjacency_matrix(G))

    # On a TU graph every node's ball overlaps every other's, so sharing one
    # quad_cache across the graph's nodes is a large win. On a citation LCC
    # it would be the opposite: millions of quads per node times thousands of
    # nodes, so there each node starts from an empty cache.
    quad_cache = {} if multi_graph else None

    for node in range(G.number_of_nodes()):
        progress.update()
        if (graph_id, node) in completed:
            continue

        # seed per nodo: due nodi diversi non devono ricevere la stessa
        # sequenza di scelte casuali, ma la corsa resta riproducibile. Il
        # passo di 1000 fra un grafo e il successivo assume grafi da meno di
        # 1000 nodi -- vero per tutti i TU qui (NCI1 arriva a 111), e per i
        # dataset a grafo singolo graph_id e' sempre 0
        profile = KL_score(G, node, quad_cache if multi_graph else {},
                           params["k"], params["temperature"],
                           params["geometric_temperature"], dist_matrix,
                           strategy=args.strategy, m=args.m,
                           seed=args.seed + 1000 * graph_id + node)

        row = [graph_id, node, profile] if multi_graph else [node, profile]
        with open(output_file, 'a', newline='') as f:
            csv.writer(f).writerow(row)

progress.close()
print(f"Wrote {output_file} (+ {os.path.basename(meta_file)})")
