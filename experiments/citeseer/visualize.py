# =============================================================
# K-HOP NEIGHBORHOOD ANALYSIS ON CITESEER (LARGEST CONNECTED COMPONENT)
# =============================================================
# This script:
#   1. Loads CiteSeer and restricts it to its largest connected component
#   2. Verifies the transform worked
#   3. Computes k-hop reachability for a single node
#   4. Computes k-hop reachability averaged over ALL nodes
#   5. Visualizes:
#       - the growth curve for a single node
#       - the growth curve averaged over all nodes (with std)
#       - the actual k-hop subgraph around a chosen node
# =============================================================


# =============================================================
# IMPORTS
# =============================================================
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data


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
# STEP 3: Compute k-hop reachable node counts for ONE node
# =============================================================
def compute_khop_counts_single_node(source_node, edge_index, max_k):
    """
    For a single source node, compute how many nodes are reachable
    within k hops, for k = 0, 1, ..., max_k.

    Returns a list of length (max_k + 1):
        counts[k] = number of nodes reachable within k hops
    """
    counts = []
    for k in range(max_k + 1):
        subset, _, _, _ = k_hop_subgraph(
            node_idx=source_node,
            num_hops=k,
            edge_index=edge_index,
            relabel_nodes=False
        )
        counts.append(subset.numel())
    return counts


# =============================================================
# STEP 4: Compute k-hop reachable node counts for ALL nodes
# =============================================================
def compute_khop_counts_all_nodes(num_nodes, edge_index, max_k, verbose=True):
    """
    Loop over every node in the graph and compute its k-hop
    reachability counts using compute_khop_counts_single_node().

    Returns a numpy array of shape (num_nodes, max_k + 1).
    """
    reachable_counts = np.zeros((num_nodes, max_k + 1), dtype=int)

    for source_node in range(num_nodes):
        reachable_counts[source_node] = compute_khop_counts_single_node(
            source_node, edge_index, max_k
        )
        if verbose and source_node % 500 == 0:
            print(f"  Processed node {source_node}/{num_nodes}")

    return reachable_counts


# =============================================================
# STEP 5: Summarize (average + std) across all nodes
# =============================================================
def summarize_khop_counts(reachable_counts):
    """
    Given the (num_nodes, max_k + 1) matrix of reachability counts,
    compute the average and std across all nodes for each k.

    Returns a pandas DataFrame with columns:
        k_hops, avg_nodes_reachable, std_nodes_reachable
    """
    max_k = reachable_counts.shape[1] - 1
    avg_reachable = reachable_counts.mean(axis=0)
    std_reachable = reachable_counts.std(axis=0)

    summary_df = pd.DataFrame({
        "k_hops": list(range(max_k + 1)),
        "avg_nodes_reachable": avg_reachable,
        "std_nodes_reachable": std_reachable,
    })
    return summary_df


# =============================================================
# STEP 6: Plot average growth curve (across all nodes)
# =============================================================
def plot_average_growth(summary_df, title="Average k-hop neighborhood growth", save_path=None):
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        summary_df["k_hops"],
        summary_df["avg_nodes_reachable"],
        yerr=summary_df["std_nodes_reachable"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("k (number of hops)")
    plt.ylabel("Average number of nodes reachable")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


# =============================================================
# STEP 7: Plot growth curve for ONE specific node
# =============================================================
def plot_single_node_growth(source_node, counts, title=None, save_path=None):
    """
    counts: list of length (max_k + 1), from compute_khop_counts_single_node()
    """
    max_k = len(counts) - 1
    if title is None:
        title = f"k-hop neighborhood growth from node {source_node}"

    plt.figure(figsize=(6, 4))
    plt.plot(range(max_k + 1), counts, marker="o")
    plt.xlabel("k (number of hops)")
    plt.ylabel("Number of nodes reachable")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


# =============================================================
# STEP 8: Visualize the ACTUAL subgraph around one node
# =============================================================
def visualize_khop_subgraph(source_node, edge_index, k, title=None, save_path=None):
    """
    Draws the k-hop subgraph around `source_node` using networkx.
    The source node is highlighted in red.
    """
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=source_node,
        num_hops=k,
        edge_index=edge_index,
        relabel_nodes=True
    )

    sub_data = Data(edge_index=edge_index_sub, num_nodes=subset.numel())
    G_sub = to_networkx(sub_data, to_undirected=True)

    source_node_new_idx = mapping.item()
    node_colors = [
        "red" if i == source_node_new_idx else "lightblue"
        for i in range(subset.numel())
    ]

    if title is None:
        title = f"{k}-hop subgraph around node {source_node} ({subset.numel()} nodes)"

    plt.figure(figsize=(7, 7))
    pos = nx.spring_layout(G_sub, seed=42)
    nx.draw(
        G_sub, pos,
        node_color=node_colors,
        node_size=150,
        edge_color="gray",
        with_labels=False
    )
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    plt.show()


# =============================================================
# MAIN
# =============================================================
def main():
    # ---- Config ----
    max_k = 10
    source_node = 0          # node to inspect individually
    subgraph_k = 4           # how many hops to draw explicitly

    # ---- Step 1 & 2: Load + verify largest connected component ----
    citeseer, data = load_citeseer_largest_cc()
    verify_largest_cc(data)

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    print(f"Working graph: {num_nodes} nodes, {edge_index.shape[1]} directed edges\n")

    # ---- Step 3 & 7: Single-node analysis ----
    print(f"Computing k-hop counts for single node {source_node}...")
    single_counts = compute_khop_counts_single_node(source_node, edge_index, max_k)
    print(f"Node {source_node} reachable counts per k: {single_counts}\n")
    plot_single_node_growth(source_node, single_counts)

    # ---- Step 8: Draw actual subgraph around that node ----
    visualize_khop_subgraph(source_node, edge_index, k=subgraph_k)

    # ---- Step 4, 5, 6: Graph-wide analysis (average over ALL nodes) ----
    print("Computing k-hop counts for ALL nodes (this may take a bit)...")
    reachable_counts = compute_khop_counts_all_nodes(num_nodes, edge_index, max_k)
    summary_df = summarize_khop_counts(reachable_counts)

    print("\nSummary (averaged over all nodes):")
    print(summary_df)

    plot_average_growth(summary_df, title="Average k-hop growth (CiteSeer, largest CC)")


if __name__ == "__main__":
    main()