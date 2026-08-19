"""
Build a supervised dataset for the "can a GNN learn local hyperbolicity?"
experiment: a list of synthetic graphs where every node carries its
KL_score as a regression target.

Node features are deliberately cheap and structural (degree, constant 1),
so that what the GNN can extract is limited to what message passing can
actually compute locally -- that is the point of the experiment (see
local_hyperbolicity_valutazione.md Sec. 6).

Two families are provided, so a model can be trained on one and tested on
the other:
    hierarchical  balanced trees plus a few random shortcut edges between
                  leaves -- the family where naive learning appears to work
    random        erdos-renyi / geometric / sbm graphs -- the family where
                  it collapses to predicting the mean

    python experiments/learning/generate_dataset.py --family hierarchical --num-graphs 50
    python experiments/learning/generate_dataset.py --family random --num-graphs 50
"""

import argparse
import os
import random
import sys

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.graphs.utils import create_graph
from src.optimization.local import KL_score
from src.optimization.objectives import compute_distance_nodes
from src.utils.config import REPO_ROOT

# graph shapes to draw from, per family. Sizes are kept under ~300 nodes so
# that the exact quad enumeration behind KL_score stays cheap.
FAMILIES = {
    "hierarchical": [
        {"type": "hierarchical", "tree_height": 4, "leaves_per_node": 2, "random_edges_added": 3},
        {"type": "hierarchical", "tree_height": 3, "leaves_per_node": 3, "random_edges_added": 4},
        {"type": "hierarchical", "tree_height": 3, "leaves_per_node": 4, "random_edges_added": 6},
        {"type": "tree", "tree_height": 4, "leaves_per_node": 2},
        {"type": "tree_of_cycles", "cycle_size": 8, "n_cycles": 4,
         "leaves_per_node": 2, "tree_height": 2},
    ],
    "random": [
        {"type": "erdos_renyi", "n": 40, "p": 0.15},
        {"type": "erdos_renyi", "n": 60, "p": 0.10},
        {"type": "geometric", "n": 50, "geometric_radius": 0.3},
        {"type": "sbm", "sizes": [25, 25], "p_intra": [0.3, 0.3], "p_inter": 0.02},
        {"type": "cycle", "n": 30},
    ],
}


def build_graph_data(graph_cfg, seed, k, temperature, geometric_temperature):
    """One synthetic graph as a PyG Data object, with y[i] = KL_score(i)."""
    G, _ = create_graph(seed=seed, **graph_cfg)

    # KL_score needs shortest paths between every pair, so drop everything
    # outside the largest component and relabel to 0..n-1
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G)

    dist_matrix, index = compute_distance_nodes(G)
    quad_cache = {}
    scores = np.array([
        KL_score(G, index[v], quad_cache, k, temperature, geometric_temperature,
                 dist_matrix)[0]
        for v in G.nodes()
    ])

    degrees = np.array([d for _, d in G.degree()], dtype=np.float32)
    x = torch.tensor(np.stack([degrees, np.ones_like(degrees)], axis=1))

    # PyG wants both directions of every undirected edge
    edges = list(G.edges())
    edge_index = torch.tensor([[u, v] for u, v in edges] + [[v, u] for u, v in edges],
                              dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index,
                y=torch.tensor(scores, dtype=torch.float32)[:, None])


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--family", choices=sorted(FAMILIES), default="hierarchical")
parser.add_argument("--num-graphs", type=int, default=50)
parser.add_argument("--k", type=int, default=4, help="Neighborhood radius for KL_score.")
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--geometric-temperature", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default=None,
                    help="Defaults to data/dataset/<family>.pt")
args = parser.parse_args()

out_path = args.out or os.path.join(REPO_ROOT, "data", "dataset", f"{args.family}.pt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

rng = random.Random(args.seed)
shapes = FAMILIES[args.family]

dataset = []
for i in tqdm(range(args.num_graphs), desc=f"generating {args.family}"):
    data = build_graph_data(rng.choice(shapes), seed=args.seed + i, k=args.k,
                            temperature=args.temperature,
                            geometric_temperature=args.geometric_temperature)
    dataset.append(data)

targets = torch.cat([d.y for d in dataset]).numpy()
print(f"\n{len(dataset)} graphs, {targets.size} nodes total")
print(f"target G*(v): mean={targets.mean():.4f}  std={targets.std():.4f}  "
      f"range [{targets.min():.4f}, {targets.max():.4f}]")

torch.save(dataset, out_path)
print(f"Saved to {out_path}")
