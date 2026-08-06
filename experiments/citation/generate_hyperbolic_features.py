"""
Compute per-node local hyperbolicity scores (KL_score) on the Cora or
CiteSeer citation network, restricted to its largest connected component.
Writes results incrementally to <dataset>_node_metrics.csv next to this
script, resuming from that file if it already exists.

Usage:
    python generate_hyperbolic_features.py --dataset cora
    python generate_hyperbolic_features.py --dataset citeseer
"""

import argparse
import os
import csv
import sys
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.optimization.local import KL_score

from scipy.sparse.csgraph import shortest_path

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import to_networkx

from tqdm import tqdm

# Planetoid's expected `name=` argument, and the KL_score hyperparameters
# tuned separately for each dataset.
DATASET_INFO = {
    "cora": {
        "name": "Cora",
        "k": 4,
        "temperature": 0.1,
        "geometric_temperature": np.arange(0.1, 2.6, 0.1),
    },
    "citeseer": {
        "name": "CiteSeer",
        "k": 8,
        "temperature": 0.1,
        "geometric_temperature": np.arange(0.05, 2.51, 0.05),
    },
}


def load_largest_cc(dataset_name):
    """
    Loads the dataset and applies the LargestConnectedComponents transform,
    so that the returned graph contains only nodes belonging to its largest
    connected component.
    """
    largest_cc = LargestConnectedComponents()
    ds = Planetoid(root="data", name=dataset_name, transform=largest_cc)
    data = ds[0]
    return ds, data


def verify_largest_cc(dataset_name, data):
    """
    Compares the transformed graph against the raw (untransformed) graph,
    and checks that the transformed graph is indeed a single connected
    component.
    """
    ds_raw = Planetoid(root="data", name=dataset_name)  # no transform
    data_raw = ds_raw[0]

    print("=" * 60)
    print("VERIFYING LargestConnectedComponents TRANSFORM")
    print("=" * 60)
    print(f"Raw graph:                    {data_raw.num_nodes} nodes, "
          f"{data_raw.edge_index.shape[1]} directed edges")
    print(f"Largest connected component: {data.num_nodes} nodes, "
          f"{data.edge_index.shape[1]} directed edges")

    G_check = to_networkx(data, to_undirected=True)
    num_components = nx.number_connected_components(G_check)
    print(f"Number of connected components in transformed graph: {num_components}")

    assert num_components == 1, \
        "The graph is NOT fully connected — transform did not work as expected!"
    print("Confirmed: graph is a single connected component.\n")


def build_nx_graph_from_data(data):
    """
    Converts the (already largest-CC, PyG) `data` object into a NetworkX
    graph with nodes relabeled to consecutive integers starting from 0.
    """
    G = to_networkx(data, to_undirected=True)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cora", "citeseer"], default="cora")
    args = parser.parse_args()

    info = DATASET_INFO[args.dataset]
    output_file = os.path.join(os.path.dirname(__file__), f"{args.dataset}_node_metrics.csv")

    # ---- Load + verify largest connected component ----
    ds, data = load_largest_cc(info["name"])
    verify_largest_cc(info["name"], data)

    # ---- Build the NetworkX graph used by KL_score ----
    G = build_nx_graph_from_data(data)
    print(f"Working graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # ---- Precompute pairwise shortest-path distances ----
    A = nx.adjacency_matrix(G)
    dist_matrix = shortest_path(A)

    k = info["k"]
    temperature = info["temperature"]
    geometric_temperature = info["geometric_temperature"]

    # ---- Resume from existing output file, if any ----
    completed_nodes = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    completed_nodes.add(int(row[0]))
    else:
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "metric_result"])

    print(f"Skipping {len(completed_nodes)} nodes already found in {output_file}")

    # ---- Loop over nodes, compute KL_score, save incrementally ----
    for node_id in tqdm(range(0, G.number_of_nodes())):
        if node_id in completed_nodes:
            continue

        result = KL_score(G, node_id, {}, k, temperature, geometric_temperature, dist_matrix)

        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([node_id, result])

        print(f"Saved node {node_id}")


if __name__ == "__main__":
    main()
