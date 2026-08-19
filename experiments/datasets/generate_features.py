"""
Compute the per-node local hyperbolicity profile (KL_score swept over a
range of geometric temperatures) for every node of a citation network's
largest connected component, and write it to
<dataset>_node_metrics.csv next to this script.

Results are appended node by node and the script resumes from whatever
is already in the csv, so it can be stopped and restarted.

    python experiments/datasets/generate_features.py --dataset cora
    python experiments/datasets/generate_features.py --dataset citeseer

Con --strategy increasing_neighborhood il ball k-hop viene campionato
(al piu' --m vicini per nodo per hop) invece che preso tutto: su Cora
con k=4 si passa da ~40s a ~0.3s per nodo. L'output finisce in un csv
separato (<dataset>_node_metrics_m<m>.csv) per non sovrascrivere quello
gia' calcolato col ball completo. Quanto costa in accuratezza lo misura
sampling_check.py.
"""

import argparse
import csv
import os
import sys

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.optimization.local import KL_score
from src.optimization.neighborhood import get_neighborhood

from common import DATASETS, load_lcc, metrics_path, to_nx

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
}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", choices=sorted(DATASETS), default="cora")
parser.add_argument("--strategy", choices=["full_neighborhood", "increasing_neighborhood"],
                    default="full_neighborhood")
parser.add_argument("--m", type=int, default=3,
                    help="Vicini tenuti per nodo per hop (solo con "
                         "--strategy increasing_neighborhood).")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed del campionamento, cosi' la corsa e' riproducibile.")
args = parser.parse_args()

params = SCORE_PARAMS[args.dataset]
output_file = metrics_path(args.dataset)
if args.strategy == "increasing_neighborhood":
    output_file = output_file.replace(".csv", f"_m{args.m}.csv")

data, node_map = load_lcc(args.dataset)
G = to_nx(data)
assert nx.number_connected_components(G) == 1, "LCC extraction failed"

# scipy's shortest_path on the adjacency matrix is much faster here than
# the networkx BFS in src.optimization.objectives; the row/column order is
# G.nodes() == 0..n-1, i.e. the index space KL_score expects
dist_matrix = shortest_path(nx.adjacency_matrix(G))

# resume: skip whatever is already in the csv
completed = set()
if os.path.exists(output_file):
    with open(output_file) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                completed.add(int(row[0]))
    print(f"Resuming: {len(completed)} of {G.number_of_nodes()} nodes already done")
else:
    with open(output_file, 'w', newline='') as f:
        csv.writer(f).writerow(["node_id", "metric_result"])

strategy_note = (f"{args.strategy} (m={args.m}, seed={args.seed})"
                 if args.strategy == "increasing_neighborhood" else args.strategy)
print(f"k={params['k']}  temperature={params['temperature']}  "
      f"T_geom: {len(params['geometric_temperature'])} values in "
      f"[{params['geometric_temperature'][0]:.2f}, "
      f"{params['geometric_temperature'][-1]:.2f}]  {strategy_note}")

for node in tqdm(range(G.number_of_nodes())):
    if node in completed:
        continue

    neighborhood = get_neighborhood(G, node, params["k"], strategy=args.strategy,
                                    m=args.m, seed=args.seed + node)
    full = get_neighborhood(G, node, params["k"])
    print(f"nodo {node}: vicini {len(neighborhood)} (full {len(full)})")

    # seed per nodo: due nodi diversi non devono ricevere la stessa
    # sequenza di scelte casuali, ma la corsa resta riproducibile
    profile = KL_score(G, node, {}, params["k"], params["temperature"],
                       params["geometric_temperature"], dist_matrix,
                       strategy=args.strategy, m=args.m,
                       seed=args.seed + node)

    with open(output_file, 'a', newline='') as f:
        csv.writer(f).writerow([node, profile])

print(f"Wrote {output_file}")
