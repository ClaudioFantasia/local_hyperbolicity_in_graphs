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
from optimization.local import precompute_energies, score_KL_divergence
import random 

def compute_scores(nodes,dist_matrix,quad_cache,k,temperature,geometry_temperature=1):
    scores = np.array([score_KL_divergence(n, dist_matrix, quad_cache, k, temperature, geometry_temperature) for n in tqdm.tqdm(nodes, desc="Computing local scores")])
    return scores

def main():
    temperature, lambda_reg, k, target = load_optimization_config()
    graph_cfg = load_graph_config()
    G, pos = create_graph(**graph_cfg)
    graph_type = graph_cfg['type']


    G = hierarchical(k = 2, depth=4, shortcuts=3)
    pos = draw_layout(G)
    draw_graphs(G,pos)
    from src.hyperbolicity.gromov import compute_gromov_hyperbolicity
    max_delta,_ = compute_gromov_hyperbolicity(G)
    print(f"The maximum of deltas is: {max_delta}")

    method_name = "KL_divergence"
    save_dir = os.path.join('data', 'latex', graph_type, method_name)
    os.makedirs(save_dir, exist_ok=True)
    tau = 0.1 # filtration constant
    geometry_temperature = 1

    # --- Precompute ---
    nodes = sorted(G.nodes())
    quad_cache = {}
    filtration_scores = np.full(fill_value = -1, shape = len(nodes))
    dist_matrix = compute_distance_nodes(G)
    print(f"The diameter of the graph is {nx.diameter(G)}")
    #quad_energy = precompute_energies(nodes, dist_matrix)

    # --- Main loop ---
    for k in range(nx.diameter(G) + 1):
    #for k in range(50,51):
        scores = compute_scores(nodes, dist_matrix, quad_cache, k, temperature)

        mask = (scores > tau) & (filtration_scores == -1)
        filtration_scores[mask] = k

        if np.all(filtration_scores != -1):
            break
    


    print(len(quad_cache.keys()))
    print("versus theoretical")
    print(math.comb(len(nodes), 4))
    save_path = f'{temperature}_beta_filtrations_cost_function.png'
    # --- Visualize ---
    draw_graph_with_values(G, pos, scores, title=f"Local Hyperbolicity Heatmap ({method_name})", save_path=save_path)
    save_path = f'{temperature}_beta_filtrations_score.png'
    draw_graph_with_values(G, pos, filtration_scores, title=f"Filtration K value ({method_name})", save_path=save_path)
    plot_hist(scores, title=f"Local Hyperbolicity (mean={scores.mean():.8f}, var={scores.var():.8f})", bins=20)

if __name__ == "__main__":
    main()
