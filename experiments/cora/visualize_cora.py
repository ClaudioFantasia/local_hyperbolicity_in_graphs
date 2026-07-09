import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_networkx, to_undirected
import numpy as np

def visualize_khop_neighborhood(G, center_node, k=2, figsize=(10, 8), seed=42):
    """
    Visualize the k-hop neighborhood of a given node in graph G.

    Parameters
    ----------
    G : networkx.Graph
        The full graph (e.g. Cora graph from load_cora()).
    center_node : int
        The node whose neighborhood we want to visualize.
    k : int
        Number of hops to include around the center node.
    figsize : tuple
        Size of the matplotlib figure.
    seed : int
        Seed for the layout algorithm (for reproducibility).
    """
    if center_node not in G:
        raise ValueError(f"Node {center_node} not in graph")

    # Get all nodes within k hops of center_node (includes center_node itself)
    khop_nodes = nx.single_source_shortest_path_length(G, center_node, cutoff=k)
    subG = G.subgraph(khop_nodes.keys()).copy()

    # Color nodes by their distance (hop count) from center_node
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

    # Highlight the center node
    nx.draw_networkx_nodes(
        subG, pos,
        nodelist=[center_node],
        node_color="red",
        node_size=500,
    )

    nx.draw_networkx_labels(subG, pos, font_size=7)

    plt.colorbar(nodes, label="Hop distance from center node")
    plt.title(f"{k}-hop neighborhood of node {center_node} "
              f"({subG.number_of_nodes()} nodes, {subG.number_of_edges()} edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return subG

def diameter_ignore_disconnected(G):
    if G.is_directed():
        components = nx.weakly_connected_components(G)
    else:
        components = nx.connected_components(G)

    max_ecc = 0
    for comp in components:
        subG = G.subgraph(comp)
        if len(subG) > 1:
            ecc = nx.eccentricity(subG)
            max_ecc = max(max_ecc, max(ecc.values()))
    return max_ecc

def build_nx_graph(edge_index, num_nodes):
    """Build an undirected NetworkX graph from a PyG edge_index."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)
    return G

def load_cora():
    """
    Download Cora and load it as a networkX graph
    """
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0]

    # Make edges undirected before splitting
    data.edge_index = to_undirected(data.edge_index)
    
    G = build_nx_graph(data.edge_index, data.num_nodes)
    return G 


def visualize_full_cora(G, labels=None, figsize=(14, 14), method="forceatlas2", seed=42):
    """
    Visualize the full Cora graph in a sparse, readable way.

    Parameters
    ----------
    G : networkx.Graph
    labels : array-like or None
        Optional per-node class labels (e.g. data.y.numpy()) for coloring.
    method : "forceatlas2" | "sfdp" | "spring"
        Layout algorithm to use.
    """
    print(f"Computing layout ({method}) for {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges...")

    if method == "forceatlas2":
        pos = nx.forceatlas2_layout(G, seed=seed)  # networkx >= 3.2
    elif method == "sfdp":
        pos = nx.nx_pydot.graphviz_layout(G, prog="sfdp")  # needs pydot + graphviz
    else:
        pos = nx.spring_layout(G, seed=seed, k=1/np.sqrt(G.number_of_nodes()))

    nodes = list(G.nodes())
    xy = np.array([pos[n] for n in nodes])

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges as a single LineCollection (fast for many edges)
    edge_lines = [(pos[u], pos[v]) for u, v in G.edges()]
    from matplotlib.collections import LineCollection
    lc = LineCollection(edge_lines, colors="gray", linewidths=0.2, alpha=0.3)
    ax.add_collection(lc)

    # Draw nodes as scatter
    if labels is not None:
        labels = np.asarray(labels)[nodes] if not isinstance(labels, dict) else \
                 np.array([labels[n] for n in nodes])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab10",
                         s=8, linewidths=0, alpha=0.85)
        plt.colorbar(sc, label="Class label", shrink=0.7)
    else:
        ax.scatter(xy[:, 0], xy[:, 1], c="steelblue", s=8, linewidths=0, alpha=0.85)

    ax.set_title(f"Cora citation graph ({G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()



# Example usage:
G = load_cora()
#diam = diameter_ignore_disconnected(G)
#print(diam)
sub = visualize_khop_neighborhood(G, center_node=1075, k=2)
#1353
exit()
# If you have the class labels from the PyG dataset:
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]
labels = data.y.numpy()  # shape (num_nodes,), index-aligned with node ids 0..N-1

visualize_full_cora(G, labels=labels, method="forceatlas2")