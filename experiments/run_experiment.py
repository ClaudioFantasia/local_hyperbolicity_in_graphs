import os
import sys
import time
import networkx as nx
import numpy as np
from tqdm import tqdm
import math
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graphs.utils import compute_distance_nodes, create_graph
from src.graphs.visualization import draw_graph_with_values, plot_hist, draw_layout, draw_graphs
from src.utils.config import load_optimization_config, load_graph_config
from src.optimization.local import KL_score

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import LargestConnectedComponents
from torch_geometric.utils import k_hop_subgraph, to_networkx

def load_dataset_largest_cc(name):
    """
    Loads the CiteSeer dataset and applies the LargestConnectedComponents
    transform, so that the returned graph contains only nodes belonging
    to its largest connected component.
    """
    name = name.lower()
    if name not in ('cora', 'citeseer'):
        raise ValueError(f"Unsupported dataset: {name}. Use 'cora' or 'citeseer'.")
    
    largest_cc = LargestConnectedComponents()
    citeseer = Planetoid(root="data", name=name, transform=largest_cc)
    data = citeseer[0]
    return data

def get_khop_subgraph_nx(dataset_name, v, k, root='./data', relabel_nodes=False):
    """
    Builds the k-hop subgraph around node v from Cora or CiteSeer
    and returns it as a NetworkX graph.

    Parameters
    ----------
    dataset_name : str
        'cora' or 'citeseer'
    v : int
        Target node index (original indexing in the dataset)
    k : int
        Number of hops
    root : str
        Where to download/cache the dataset
    relabel_nodes : bool
        If True, subgraph nodes are labeled 0..n-1 (local indices).
        If False (default), nodes keep their original dataset indices.

    Returns
    -------
    G_sub : networkx.Graph
        The k-hop subgraph, with node attribute 'y' (label).
    """
    data = load_dataset_largest_cc(dataset_name)

    # Always relabel internally: this guarantees edge_index, x, y and
    # num_nodes are all consistent and correctly sized.
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=v,
        num_hops=k,
        edge_index=data.edge_index,
        relabel_nodes=True,
        num_nodes=data.num_nodes
    )

    sub_x = data.x[subset]
    sub_y = data.y[subset]

    sub_data = Data(x=sub_x, edge_index=edge_index, y=sub_y,
                     num_nodes=subset.size(0))

    G_sub = to_networkx(sub_data, node_attrs=['y'], to_undirected=True)

    #if not relabel_nodes:
        # Map local indices (0..n-1) back to original dataset node ids
        #local_to_original = {i: int(subset[i]) for i in range(subset.size(0))}
        #G_sub = nx.relabel_nodes(G_sub, local_to_original)

    return G_sub




def main():
    parser = argparse.ArgumentParser(description="Run Local Hyperbolicity Experiments")
    parser.add_argument("--method", type=str, required=True, 
                        choices=["max_pooling", "softmax", "KL_divergence", "entropic"],
                        help="The aggregation method to use.")
    args = parser.parse_args()
    method_name = args.method

    if method_name != "entropic":
        print("Sorry the other methods are not re-implemented yet")
        raise KeyboardInterrupt()

    # Load configuration
    temperature, geometric_temperature, lambda_reg, sigma, k, target = load_optimization_config()
    #geometric_temperature = np.array([1.0])
    graph_cfg = load_graph_config()
    G, pos = create_graph(**graph_cfg)
    from src.optimization.local import get_neighborhood
    m = 2
    k = 50
    #G = get_khop_subgraph_nx(dataset_name='citeseer', v=0, k=50)

    #ret = get_neighborhood(G=G,target=10,k=50,strategy='increasing_neighborhood',m=m)
    #print(len(ret))
    #print(ret)
    #exit()

    #graph_type = 'citeseer'
    
    #graph_type = graph_cfg['type']
    pos = draw_layout(G)
    draw_graphs(G, pos)
    #exit()

    #print(f"Graph diameter: {nx.diameter(G)}")
    if not nx.is_connected(G):
        print("graph not connected, lets stop here")
        raise KeyboardInterrupt()
    
    #save_dir = os.path.join('data', 'figures', graph_type, method_name)
    #os.makedirs(save_dir, exist_ok=True)

    nodes = sorted(G.nodes())
    t0 = time.perf_counter()
    dist_matrix = compute_distance_nodes(G)
    quad_cache = {}
    #strategy = "increasing_neighborhood"
    strategy = 'full_neighborhood'
    scores = np.array([KL_score(G,v,quad_cache,k,temperature,geometric_temperature,dist_matrix,strategy,m) for v in range(len(G.nodes()))])
    elapsed = time.perf_counter() - t0
    print(f"[{method_name}] {k}-hop done in {elapsed:.2f}s")
    print(f"Cache size: {len(quad_cache)} vs theoretical max {math.comb(len(nodes), 4)}")
    plot_hist(scores)

    # Visualize
    #print(geometric_temperature)
  
    draw_graph_with_values(G, pos, scores, title=f"Local Hyperbolicity Heatmap ({method_name})", save_path=None)

if __name__ == "__main__":
    main()