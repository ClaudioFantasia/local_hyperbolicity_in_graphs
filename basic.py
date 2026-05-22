import itertools
import networkx as nx
import numpy as np
from tqdm import tqdm
import yaml 
from src.graphs.utils import compute_distance_nodes, create_SBM_graph
from src.graphs.visualization import draw_graphs, draw_graph_with_values, plot_hist, draw_layout
from src.optimization.objectives import gromov_energy
from scipy.special import softmax



def load_config(path="configs/optimization_parameters.yaml"):
    cfg = yaml.safe_load(open(path))["optimization"]
    return cfg["temperature"], cfg["lambda_reg"]

# --- Local score ---
def local_score(target, dist_matrix, k, temperature):
    neighborhood = np.where((dist_matrix[target] <= k) & (dist_matrix[target] > 0))[0]
    quads = [(target, x, y, z) for x, y, z in itertools.combinations(neighborhood, 3)]
    if not quads:
        return 0.0

    e_gromov = np.array(gromov_energy(quads, dist_matrix))
    #dists    = dist_matrix[target, np.array(list(q[1:] for q in quads))]
    #weights  = softmax(-np.sum(dists, axis=1) / temperature)
    return float(np.max(e_gromov))


# --- Config & graph ---
temperature, lambda_reg = load_config()
k = 4

G = create_SBM_graph(sizes=[35,35],p_intra=[0.4,0.4],p_inter=0.001)
pos = draw_layout(G)

G = nx.random_geometric_graph(n=150, radius=0.2, seed=4)
pos = {n: d["pos"] for n, d in G.nodes(data=True)}

draw_graphs(G, pos)
dist_matrix = compute_distance_nodes(G)

# --- Main loop ---
nodes  = sorted(G.nodes())
scores = np.array([local_score(n, dist_matrix, k, temperature) for n in tqdm(nodes, desc="Computing local scores")])

# --- Visualize ---
draw_graph_with_values(G, pos, scores, title="Local Hyperbolicity Heatmap")
plot_hist(scores, title=f"Local Hyperbolicity (mean={scores.mean():.4f}, var={scores.var():.4f})", bins=20)