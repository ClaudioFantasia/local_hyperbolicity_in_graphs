import os
import sys
import time
import networkx as nx
import numpy as np
import tqdm
import itertools
import math 


from src.graphs.utils import compute_distance_nodes, create_graph
from src.graphs.visualization import draw_graphs, draw_graph_with_values, plot_hist, draw_layout, draw_quadruples
from src.utils.config import load_optimization_config, load_graph_config
from src.hyperbolicity.local import precompute_energies, score_KL_divergence
from scipy.special import softmax
from src.optimization.objectives import gromov_energy, normalize
from src.optimization.solver import solve_KL_regularization
import random

import networkx as nx
import matplotlib.pyplot as plt
def hierarchical(k: int = 3, depth: int = 3, shortcuts: int = 6,
                 seed: int = None) -> nx.Graph:
    """
    Albero k-ario di profondità d con shortcut casuali tra foglie.

    L'albero è perfettamente iperbolico (δ = 0), gli shortcut aumentano
    δ in modo controllato. Permette di studiare come i cicli degradano
    l'iperbolicità. Alzare 'shortcuts' da 0 gradualmente per calibrare.

    Parametri:
        k        : grado (numero di figli per nodo interno)
        depth    : profondità dell'albero
        shortcuts: archi casuali aggiunti tra foglie
    """
    rng = random.Random(seed)
    G = nx.Graph()
    idx = [0]
    leaves = []

    def build(parent, d):
        if d == 0:
            leaves.append(parent)
            return
        for _ in range(k):
            child = idx[0] + 1
            idx[0] = child
            G.add_node(child)
            G.add_edge(parent, child)
            build(child, d - 1)

    G.add_node(0)
    build(0, depth)

    for _ in range(shortcuts):
        a = rng.choice(leaves)
        b = rng.choice(leaves)
        if a != b:
            G.add_edge(a, b)

    return G

def create_molecule_like_graph():
    # 1. Create the base components
    tree_part = nx.balanced_tree(r=2, h=3)  # 15 nodes
    main_path = nx.path_graph(16)           # Long backbone path (16 nodes)
    cycle_part = nx.cycle_graph(8)          # 8 nodes
    grid_part = nx.grid_2d_graph(4, 4)      # 16 nodes
    
    # 2. Join all of them disjointly to ensure sequential scalar IDs
    G = nx.disjoint_union_all([tree_part, main_path, cycle_part, grid_part])
    
    # 3. Calculate starting indices for each component in the merged graph
    tree_start = 0
    path_start = len(tree_part)
    cycle_start = path_start + len(main_path)
    grid_start = cycle_start + len(cycle_part)
    
    # 4. Wire the "Molecule" structure together
    
    # A. Connect Tree to the beginning of the Main Path
    G.add_edge(tree_start, path_start)
    
    # B. Connect Grid to the end of the Main Path
    path_end = path_start + len(main_path) - 1
    G.add_edge(path_end, grid_start)
    
    # C. Branch the Cycle off the center of the Main Path
    path_middle = path_start + (len(main_path) // 2)
    G.add_edge(path_middle, cycle_start)
    
    return G




def main():
    temperature, lambda_reg, k, target = load_optimization_config()

    graph_cfg = load_graph_config()
    G, pos = create_graph(**graph_cfg)
  
    #G = create_molecule_like_graph()
    #pos = draw_layout(G)
    
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
    for j in range(len(G.nodes())):
        print(j,dist_matrix[0,j])
    quad_cache = precompute_energies(nodes, dist_matrix)
    top_items = sorted(
        quad_cache.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    for quad, value in top_items:
        print(f"{quad}: {value}")

    
    
    # --- Main loop ---
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quads = list(itertools.combinations(neighborhood, 4))

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
    save_path = f'{target}_{k}_contributions.png'
    draw_graph_with_values(G, pos, node_contributions, title=f"Mu Heatmap with target:({target})", save_path=None)
    save_path = f'{target}_{k}_mu.png'
    plot_hist(mu, title=f"Mu distribution (mean={mu.mean():.8f}, var={mu.var():.8f})", bins=20, save_path=None)
    scores = np.array([score_KL_divergence(n, dist_matrix, quad_cache, k, temperature) for n in tqdm.tqdm(nodes, desc="Computing local scores")])
    save_path = f'{k}_hop_{temperature}_beta_cost_function.png'
    draw_graph_with_values(G, pos, scores, title=f"Local Hyperbolicity Heatmap ({method_name})", save_path=save_path)
    plot_hist(scores, title=f"Local Hyperbolicity (mean={scores.mean():.8f}, var={scores.var():.8f})", bins=20)


if __name__ == "__main__":
    main()
