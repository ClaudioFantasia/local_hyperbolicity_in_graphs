import os
import sys
import time
import networkx as nx
import numpy as np
import tqdm
import itertools
import math 


from src.graphs.utils import compute_distance_nodes, create_graph
from src.graphs.visualization import draw_graphs, draw_graph_with_values, plot_hist, draw_layout
from src.utils.config import load_optimization_config, load_graph_config
from src.hyperbolicity.local import precompute_energies, score_KL_divergence
from scipy.special import softmax
from src.optimization.objectives import gromov_energy, normalize
from src.optimization.solver import solve_KL_regularization

def main():
    temperature, lambda_reg, k, target = load_optimization_config()

    graph_cfg = load_graph_config()
    G, pos = create_graph(**graph_cfg)
    print(nx.diameter(G))

    graph_type = graph_cfg['type']
    method_name = "KL_divergence"
    save_dir = os.path.join('data', 'newFig', graph_type, method_name)
    os.makedirs(save_dir, exist_ok=True)
    geometry_temperature = 1
    # --- Precompute ---
    nodes = sorted(G.nodes())
    quad_cache = {}
    t0 = time.perf_counter()
    dist_matrix = compute_distance_nodes(G)
    
    #quad_energy = precompute_energies(nodes, dist_matrix)
    # --- Main loop ---
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quads = list(itertools.combinations(neighborhood, 4))
    if not quads:
        return 0.0
    energies = gromov_energy(quads, dist_matrix)
    quad_cache.update(zip(quads, energies))

    w_local = np.mean(dist_matrix[np.array(quads), target], axis=1)
    w_local = softmax(-1 * w_local / geometry_temperature)

    e_gromov = np.array([quad_cache[q] for q in quads])

    mu = solve_KL_regularization(e_gromov, w_local, temperature)

    # Calcolo del contributo di ogni nodo ai valori di mu
    node_contributions = np.zeros(len(nodes))
    for i, q in enumerate(quads):
        for node in q:
            node_contributions[node] += mu[i]

    elapsed = time.perf_counter() - t0
    print(f"[{method_name}] {k}-hop done in {elapsed:.2f}s")

    print(len(quad_cache.keys()))
    print("versus theoretical")
    print(math.comb(len(nodes), 4))

    # --- Visualize ---
    draw_graph_with_values(G, pos, node_contributions, title=f"Mu Heatmap with target:({target})", save_path=None)
    plot_hist(mu, title=f"Local Hyperbolicity (mean={mu.mean():.8f}, var={mu.var():.8f})", bins=20)

if __name__ == "__main__":
    main()
