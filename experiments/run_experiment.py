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
from src.graphs.visualization import draw_graph_with_values, plot_hist
from src.utils.config import load_optimization_config, load_graph_config
from optimization.local import score_max, score_softmax, score_KL_divergence, score_entropic

def main():
    parser = argparse.ArgumentParser(description="Run Local Hyperbolicity Experiments")
    parser.add_argument("--method", type=str, required=True, 
                        choices=["max_pooling", "softmax", "KL_divergence", "entropic"],
                        help="The aggregation method to use.")
    args = parser.parse_args()
    method_name = args.method

    # Load configuration
    temperature, geometric_temperature, lambda_reg, k, target = load_optimization_config()
    
    graph_cfg = load_graph_config()
    G, pos = create_graph(**graph_cfg)
    graph_type = graph_cfg['type']

    save_dir = os.path.join('data', 'figures', graph_type, method_name)
    os.makedirs(save_dir, exist_ok=True)

    nodes = sorted(G.nodes())
    t0 = time.perf_counter()
    dist_matrix = compute_distance_nodes(G)
    quad_cache = {}

    print(f"Graph diameter: {nx.diameter(G)}")

    def compute_score(n):
        if method_name == "max_pooling":
            return score_max(n, dist_matrix, quad_cache, k)
        elif method_name == "softmax":
            return score_softmax(n, dist_matrix, quad_cache, k, temperature)
        elif method_name == "KL_divergence":
            return score_KL_divergence(n, dist_matrix, quad_cache, k, temperature, geometric_temperature)
        elif method_name == "entropic":
            return score_entropic(n, dist_matrix, quad_cache, k, temperature, lambda_reg)

    scores = np.array([compute_score(n) for n in tqdm(nodes, desc=f"Computing local scores ({method_name})")])
    elapsed = time.perf_counter() - t0
    print(f"[{method_name}] {k}-hop done in {elapsed:.2f}s")
    print(f"Cache size: {len(quad_cache)} vs theoretical max {math.comb(len(nodes), 4)}")

    # Visualize
    save_path = os.path.join(save_dir, f"{k}_hop_T{temperature}_L{lambda_reg}.png")
    draw_graph_with_values(G, pos, scores, title=f"Local Hyperbolicity Heatmap ({method_name})", save_path=save_path)
    
    hist_title = f"Local Hyperbolicity (mean={scores.mean():.8f}, var={scores.var():.8f})"
    plot_hist(scores, title=hist_title, bins=20)

if __name__ == "__main__":
    main()