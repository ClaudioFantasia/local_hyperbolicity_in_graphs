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
from src.hyperbolicity.local import precompute_energies, score_KL_divergence
import random 
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

def hierarchical(k: int = 3, depth: int = 3, shortcuts: int = 6,
                 seed: int = 42) -> nx.Graph:
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
