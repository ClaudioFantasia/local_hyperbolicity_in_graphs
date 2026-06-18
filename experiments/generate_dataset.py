import os
import sys
import argparse
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graphs.utils import compute_distance_nodes, create_graph
from src.hyperbolicity.local import score_KL_divergence

def generate_graph_data(graph_type, params, k=2, temperature=1.0):
    # 1. Create the graph
    G, _ = create_graph(graph_type, **params)
    
    # Check if graph is connected, otherwise score might have issues
    if not nx.is_connected(G):
        components = sorted(nx.connected_components(G), key=len, reverse=True)
        G = G.subgraph(components[0]).copy()
        G = nx.convert_node_labels_to_integers(G)
        
    num_nodes = G.number_of_nodes()
    
    # 2. Compute true labels (local hyperbolicity)
    dist_matrix = compute_distance_nodes(G)
    nodes = list(G.nodes())
    quad_cache = {}
    
    y = np.zeros((num_nodes, 1), dtype=np.float32)
    geometry_temperature = 1.0
    
    print(f"Generating scores for {graph_type} with {num_nodes} nodes...")
    for n_idx in tqdm(nodes, leave=False):
        score = score_KL_divergence(n_idx, dist_matrix, quad_cache, k, temperature, geometry_temperature)
        y[n_idx, 0] = score
        
    # 3. Create PyG Data object
    # For features, we use node degree and constant 1
    degrees = np.array([d for n, d in G.degree()]).reshape(-1, 1).astype(np.float32)
    constants = np.ones((num_nodes, 1), dtype=np.float32)
    x = torch.tensor(np.hstack([degrees, constants]), dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Edges
    edges = list(G.edges())
    # Add reverse edges to make it undirected in edge_index format
    edge_index = torch.tensor([[u, v] for u, v in edges] + [[v, u] for u, v in edges], dtype=torch.long).t().contiguous()
    
    # Build data dict
    data = Data(x=x, edge_index=edge_index, y=y_tensor)
    return data

def main():
    if not os.path.exists('data/dataset'):
        os.makedirs('data/dataset', exist_ok=True)
        
    graphs_config = [
        ('tree', {'leaves_per_node': 2, 'tree_height': 4}), # 31 nodes
        ('tree', {'leaves_per_node': 3, 'tree_height': 3}), # 40 nodes
        ('erdos_renyi', {'n': 40, 'p': 0.15}),
        ('erdos_renyi', {'n': 50, 'p': 0.2}),
        ('cycle', {'n': 30}),
        ('cycle', {'n': 40}),
        ('geometric', {'n': 40, 'geometric_radius': 0.3}),
        ('complete', {'n': 20}),
    ]
    
    dataset = []
    for gtype, params in graphs_config:
        try:
            data = generate_graph_data(gtype, params, k=2)
            dataset.append(data)
        except Exception as e:
            print(f"Error generating {gtype}: {e}")
            
    # Save dataset
    out_path = 'data/dataset/synthetic_hyperbolicity.pt'
    torch.save(dataset, out_path)
    print(f"Dataset generated and saved to {out_path} ({len(dataset)} graphs)")

if __name__ == "__main__":
    main()
