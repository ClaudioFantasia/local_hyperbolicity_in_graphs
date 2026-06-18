import os
import sys
import time
import networkx as nx
import numpy as np
import tqdm
import itertools
import math
# Aggiungo la cartella principale al path per trovare 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graphs.utils import compute_distance_nodes, create_graph
from src.graphs.visualization import draw_graphs, draw_graph_with_values, plot_hist, draw_layout
from src.utils.config import load_optimization_config, load_graph_config
from src.hyperbolicity.local import precompute_energies, score_entropic
from src.graphs.utils import make_tree_with_grid
def main():
    temperature, lambda_reg, k = load_optimization_config()
    graph_cfg = load_graph_config()
    G, pos = create_graph(**graph_cfg)
    graph_type = graph_cfg['type']
    method_name = "normalization_entropic"
    save_dir = os.path.join('data', 'lambdaEffect', graph_type, method_name)
    os.makedirs(save_dir, exist_ok=True)

    # --- Precompute ---
    nodes = sorted(G.nodes())
    quad_cache = {}
    t0 = time.perf_counter()
    dist_matrix = compute_distance_nodes(G)
    #quad_cache = precompute_energies(nodes, dist_matrix)


    # --- Main loop ---
    scores = np.array([score_entropic(n, dist_matrix, quad_cache, k, temperature, lambda_reg) for n in tqdm.tqdm(nodes, desc="Computing local scores")])
    elapsed = time.perf_counter() - t0
    print(f"[{method_name}] {k}-hop done in {elapsed:.2f}s")

    print(len(quad_cache.keys()))
    print("versus theoretical")
    print(math.comb(len(nodes), 4))

    # --- Visualize ---
    save_path = os.path.join(save_dir, f"{k}_hop_{elapsed:.2f}s_{temperature}_temp_{lambda_reg}_lambda.png")
    fixed_range_save_path = os.path.join(save_dir, f"{k}_hop_fixed_range_{elapsed:.2f}s_{temperature}_temp_{lambda_reg}_lambda.png")
    draw_graph_with_values(G, pos, scores, title=f"Local Hyperbolicity Heatmap ({method_name})", save_path=save_path)
    draw_graph_with_values(G, pos, scores, title=f"Local Hyperbolicity Heatmap ({method_name})", save_path=fixed_range_save_path, vmin = 0, vmax = 1)
    plot_hist(scores, title=f"Local Hyperbolicity (mean={scores.mean():.8f}, var={scores.var():.8f})", bins=20)

if __name__ == "__main__":
    main()
