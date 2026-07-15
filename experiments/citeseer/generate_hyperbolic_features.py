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

OUTPUT_FILE = "citeseer_node_metrics.csv"


# =============================================================
# STEP 1: Load CiteSeer, explicitly applying LargestConnectedComponents
# =============================================================
def load_citeseer_largest_cc():
    """
    Loads the CiteSeer dataset and applies the LargestConnectedComponents
    transform, so that the returned graph contains only nodes belonging
    to its largest connected component.
    """
    largest_cc = LargestConnectedComponents()
    citeseer = Planetoid(root="data", name="citeseer", transform=largest_cc)
    data = citeseer[0]
    return citeseer, data


# =============================================================
# STEP 2: Verify the transform actually took effect
# =============================================================
def verify_largest_cc(data):
    """
    Compares the transformed graph against the raw (untransformed) graph,
    and checks that the transformed graph is indeed a single connected
    component.
    """
    citeseer_raw = Planetoid(root="data", name="citeseer")  # no transform
    data_raw = citeseer_raw[0]

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


# =============================================================
# STEP 3: Convert the PyG largest-CC data into a clean networkx graph
# =============================================================
def build_nx_graph_from_data(data):
    """
    Converts the (already largest-CC, PyG) `data` object into a
    NetworkX graph with nodes relabeled to consecutive integers
    starting from 0 — mirroring how the Cora script prepares its graph.
    """
    G = to_networkx(data, to_undirected=True)
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    return G


# =============================================================
# MAIN
# =============================================================
def main():
    # ---- Step 1 & 2: Load + verify largest connected component ----
    citeseer, data = load_citeseer_largest_cc()
    verify_largest_cc(data)

    # ---- Step 3: Build the NetworkX graph used by KL_score ----
    G = build_nx_graph_from_data(data)
    print(f"Working graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

    # ---- Step 4: Precompute pairwise shortest-path distances ----
    A = nx.adjacency_matrix(G)
    dist_matrix = shortest_path(A)

    # ---- Step 5: Hyperparameters for KL_score (same as Cora script) ----
    k = 8
    temperature = 0.1
    geometric_temperature = np.arange(0.05, 2.51, 0.05)

    # ---- Step 6: Resume from existing output file, if any ----
    completed_nodes = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row:
                    completed_nodes.add(int(row[0]))
    else:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "metric_result"])

    print(f"Skipping {len(completed_nodes)} nodes already found in {OUTPUT_FILE}")

    # ---- Step 7: Loop over nodes, compute KL_score, save incrementally ----
    for node_id in tqdm(range(0, G.number_of_nodes())):
        if node_id in completed_nodes:
            continue

        result = KL_score(G, node_id, {}, k, temperature, geometric_temperature, dist_matrix)

        with open(OUTPUT_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([node_id, result])

        print(f"Saved node {node_id}")


if __name__ == "__main__":
    main()