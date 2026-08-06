"""
K-hop neighborhood analysis and visualization on Cora or CiteSeer, largest
connected component only. Select the dataset with --dataset {cora,citeseer}.

This script:
    1. Loads the dataset and restricts it to its largest connected component
    2. Verifies the transform worked
    3. Computes k-hop reachability for a single node
    4. Computes k-hop reachability averaged over ALL nodes
    5. Visualizes:
        - a single node's k-hop neighborhood, colored by hop distance
        - the growth curve for a single node
        - the growth curve averaged over all nodes (with std)
        - the actual k-hop subgraph around a chosen node

Usage:
    python visualize.py --dataset cora
    python visualize.py --dataset citeseer
"""

import argparse

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.data import Data

DATASET_NAMES = {"cora": "Cora", "citeseer": "CiteSeer"}


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


def visualize_khop_neighborhood(G, center_node, k=2, figsize=(10, 8), seed=42):
    """
    Visualize the k-hop neighborhood of a given node in a networkx graph G,
    coloring each node by its hop distance from center_node.
    """
    if center_node not in G:
        raise ValueError(f"Node {center_node} not in graph")

    khop_nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=k)
    subG = G.subgraph(khop_nodes.keys()).copy()
    distances = [khop_nodes[n] for n in subG.nodes()]

    pos = nx.spring_layout(subG, seed=seed)

    plt.figure(figsize=figsize)
    nodes = nx.draw_networkx_nodes(
        subG, pos,
        node_color=distances,
        cmap=plt.cm.viridis_r,
        node_size=300,
        linewidths=1,
        edgecolors="black",
    )
    nx.draw_networkx_edges(subG, pos, alpha=0.4)
    nx.draw_networkx_nodes(subG, pos, nodelist=[center_node], node_color="red", node_size=500)
    nx.draw_networkx_labels(subG, pos, font_size=7)

    plt.colorbar(nodes, label="Hop distance from center node")
    plt.title(f"{k}-hop neighborhood of node {center_node} "
              f"({subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return subG


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=["cora", "citeseer"], default="cora")
    parser.add_argument("--max-k", type=int, default=10, help="Max hop count for growth curves.")
    parser.add_argument("--source-node", type=int, default=0, help="Node to inspect individually.")
    parser.add_argument("--subgraph-k", type=int, default=4, help="Hops to draw explicitly around --source-node.")
    args = parser.parse_args()

    dataset_name = DATASET_NAMES[args.dataset]

    # ---- Load + verify largest connected component ----
    ds, data = load_largest_cc(dataset_name)
    verify_largest_cc(dataset_name, data)

    num_nodes = data.num_nodes
    edge_index = data.edge_index

    print(f"Working graph: {num_nodes} nodes, {edge_index.shape[1]} directed edges\n")

    # ---- Hop-distance-colored neighborhood around the source node ----
    G = to_networkx(data, to_undirected=True)
    visualize_khop_neighborhood(G, center_node=args.source_node, k=args.subgraph_k)

    # ---- Single-node growth curve ----
    print(f"Computing k-hop counts for single node {args.source_node}...")
    single_counts = compute_khop_counts_single_node(args.source_node, edge_index, args.max_k)
    print(f"Node {args.source_node} reachable counts per k: {single_counts}\n")
    plot_single_node_growth(args.source_node, single_counts)

    # ---- Highlighted subgraph around that node ----
    visualize_khop_subgraph(args.source_node, edge_index, k=args.subgraph_k)

    # ---- Graph-wide analysis (average over ALL nodes) ----
    print("Computing k-hop counts for ALL nodes (this may take a bit)...")
    reachable_counts = compute_khop_counts_all_nodes(num_nodes, edge_index, args.max_k)
    summary_df = summarize_khop_counts(reachable_counts)

    print("\nSummary (averaged over all nodes):")
    print(summary_df)

    plot_average_growth(summary_df, title=f"Average k-hop growth ({dataset_name}, largest CC)")


if __name__ == "__main__":
    main()
