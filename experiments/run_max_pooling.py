import os
import sys
import time
import networkx as nx
import numpy as np
from tqdm import tqdm
import math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graphs.utils import compute_distance_nodes, create_graph
from src.graphs.visualization import draw_graphs, draw_graph_with_values, plot_hist, draw_layout
from src.utils.config import load_optimization_config, load_graph_config
from src.hyperbolicity.local import score_max, precompute_energies

def main():
    temperature, lambda_reg, k, target = load_optimization_config()
    graph_cfg = load_graph_config()

    G, pos = create_graph(**graph_cfg)

    
    graph_type = graph_cfg['type']
    method_name = "max_pooling"
    save_dir = os.path.join('data', 'figures', graph_type, method_name)
    os.makedirs(save_dir, exist_ok=True)

    nodes = sorted(G.nodes())
    t0 = time.perf_counter()
    quad_cache = {}
    dist_matrix = compute_distance_nodes(G)
    

    

    # --- Main loop ---
    nodes = sorted(G.nodes())
    
    scores = np.array([score_max(n, dist_matrix, quad_cache, k) for n in tqdm(nodes, desc="Computing local scores")])
    elapsed = time.perf_counter() - t0
    print(f"[{method_name}] {k}-hop done in {elapsed:.2f}s")
    
    print(len(quad_cache.keys()))
    print("versus")
    print(math.comb(len(nodes), k))

    # --- Visualize ---
    save_path = os.path.join(save_dir, f"{k}_hop_{elapsed:.2f}s.png")
    draw_graph_with_values(G, pos, scores, title=f"Local Hyperbolicity Heatmap ({method_name})", save_path=save_path)
    plot_hist(scores, title=f"Local Hyperbolicity (mean={scores.mean():.8f}, var={scores.var():.8f})", bins=20)

if __name__ == "__main__":
    main()
