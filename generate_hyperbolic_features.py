import os
import csv
import time
import networkx as nx 
import numpy as np

from src.hyperbolicity.local import score_KL_divergence,score_KL_divergence_batched
from src.graphs.visualization import draw_layout, draw_graphs
from src.graphs.utils import create_graph
from src.utils.config import load_optimization_config, load_graph_config

from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix
from scipy.special import softmax, logsumexp

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import to_networkx, to_undirected

from tqdm import tqdm
TOTAL_CORA_NODES = 2708 
OUTPUT_FILE = "cora_node_metrics.csv"




def compute_metric_for_node(node_id):
    # Your custom algorithm goes here
    time.sleep(0.1) 
    return f"metric_value_{node_id}"


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


G = load_cora()
#graph_cfg = load_graph_config()
#G, pos = create_graph(**graph_cfg)
#graph_type = graph_cfg['type']
A = nx.adjacency_matrix(G)
dist_matrix = shortest_path(A)

quad_cache = {}
k = 2
temperature = 0.1
geometric_temperature = np.arange(0.1,5.5,0.25)


# 1. Check what has already been computed
completed_nodes = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # Skip header
        for row in reader:
            if row:
                completed_nodes.add(int(row[0]))
else:
    # If the file doesn't exist yet, create it and write the header
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "metric_result"])

print(f"Skipping {len(completed_nodes)} nodes already found in {OUTPUT_FILE}")

# 2. Loop and save
for node_id in tqdm(range(G.number_of_nodes())):
    # Skip if already computed
    if node_id in completed_nodes:
        continue
    if len(quad_cache) > 5_000_000:
        quad_cache = {}
    result = score_KL_divergence(node_id,dist_matrix,quad_cache,k,temperature,geometric_temperature)
    
    # Save immediately
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([node_id, result])
        
    print(f"Saved node {node_id}")