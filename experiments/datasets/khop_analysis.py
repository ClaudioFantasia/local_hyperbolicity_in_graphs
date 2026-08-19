"""
How fast do k-hop neighborhoods grow on a citation network?

This is the sanity check behind the choice of k in generate_features.py:
KL_score enumerates 4-tuples of the k-hop ball, so the ball size is what
decides whether a node is tractable at all.

Produces:
    1. the k-hop neighborhood of one node, colored by hop distance
    2. that node's growth curve (nodes reachable within k hops)
    3. the growth curve averaged over every node, with std

    python experiments/datasets/khop_analysis.py --dataset cora
    python experiments/datasets/khop_analysis.py --dataset citeseer --source-node 42
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from common import DATASETS, load_lcc, to_nx


def khop_counts(G, source, max_k):
    """counts[k] = how many nodes are within k hops of `source`,
    for k = 0..max_k."""
    lengths = nx.single_source_shortest_path_length(G, source, cutoff=max_k)
    counts = np.zeros(max_k + 1, dtype=int)
    for d in lengths.values():
        counts[d] += 1
    return np.cumsum(counts)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", choices=sorted(DATASETS), default="cora")
parser.add_argument("--max-k", type=int, default=10)
parser.add_argument("--source-node", type=int, default=0,
                    help="LCC node index to inspect individually.")
parser.add_argument("--subgraph-k", type=int, default=4,
                    help="Hops to draw explicitly around --source-node.")
args = parser.parse_args()

data, node_map = load_lcc(args.dataset)
G = to_nx(data)

# ---- the source node's k-hop neighborhood, colored by hop distance ----
hops = nx.single_source_shortest_path_length(G, args.source_node,
                                             cutoff=args.subgraph_k)
subG = G.subgraph(hops.keys())
pos = nx.spring_layout(subG, seed=42)

plt.figure(figsize=(10, 8))
nodes = nx.draw_networkx_nodes(subG, pos, node_color=[hops[n] for n in subG.nodes()],
                               cmap=plt.cm.viridis_r, node_size=120,
                               linewidths=0.5, edgecolors="black")
nx.draw_networkx_edges(subG, pos, alpha=0.4)
nx.draw_networkx_nodes(subG, pos, nodelist=[args.source_node],
                       node_color="red", node_size=300)
plt.colorbar(nodes, label="hop distance from the source node")
plt.title(f"{args.subgraph_k}-hop neighborhood of node {args.source_node} "
          f"({args.dataset}): {subG.number_of_nodes()} nodes, "
          f"{subG.number_of_edges()} edges")
plt.axis("off")
plt.tight_layout()
plt.show()

# ---- that node's growth curve ----
single = khop_counts(G, args.source_node, args.max_k)
print(f"node {args.source_node} reachable counts per k: {single.tolist()}")

plt.figure(figsize=(6, 4))
plt.plot(range(args.max_k + 1), single, marker="o")
plt.xlabel("k (hops)")
plt.ylabel("nodes reachable within k hops")
plt.title(f"k-hop growth from node {args.source_node} ({args.dataset})")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- averaged over every node ----
print(f"Computing k-hop counts for all {G.number_of_nodes()} nodes...")
all_counts = np.array([khop_counts(G, v, args.max_k) for v in G.nodes()])
avg, std = all_counts.mean(axis=0), all_counts.std(axis=0)

for k in range(args.max_k + 1):
    print(f"  k={k:>2}: {avg[k]:>8.1f} +/- {std[k]:.1f} nodes "
          f"({avg[k] / G.number_of_nodes():.1%} of the graph)")

plt.figure(figsize=(7, 5))
plt.errorbar(range(args.max_k + 1), avg, yerr=std, marker="o", capsize=4)
plt.xlabel("k (hops)")
plt.ylabel("nodes reachable within k hops")
plt.title(f"Average k-hop growth ({args.dataset}, largest CC)")
plt.grid(True)
plt.tight_layout()
plt.show()
