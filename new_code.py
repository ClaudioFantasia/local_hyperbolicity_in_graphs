import itertools, os, time
import networkx as nx
import numpy as np
import yaml

from src.graphs.utils import compute_distance_nodes, make_tree_of_cycles, make_tree_with_grid, create_SBM_graph
from src.graphs.visualization import draw_graphs, draw_graph_with_values, plot_hist, draw_layout
from src.optimization.objectives import gromov_energy, normalize
from src.optimization.solver import solve_entropic_regularization


def load_config(path="configs/optimization_parameters.yaml"):
    cfg = yaml.safe_load(open(path))["optimization"]
    return cfg["temperature"], cfg["lambda_reg"]


def precompute_energies(nodes, dist_matrix):
    quads = list(itertools.combinations(nodes, 4))
    energies = gromov_energy(quads, dist_matrix)
    print(f"Global Gromov hyperbolicity: {np.max(energies):.4f}")
    return dict(zip(quads, energies))


def local_score(target, dist_matrix, quad_energy, k, temperature, lambda_reg):
    print(f'Analyzing {target} node')
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quads = list(itertools.combinations(neighborhood, 4))
    if not quads:
        return 0.0

    e_gromov = normalize(np.array([quad_energy[q] for q in quads]))
    w_local  = normalize(np.mean(dist_matrix[np.array(quads), target], axis=1))

    cost = e_gromov - lambda_reg * w_local
    mu   = solve_entropic_regularization(cost, temperature)
    return float(np.sum(mu * cost - temperature * mu * np.log(mu)))


# --- Config & graph ---
temperature, lambda_reg = load_config()
k = 4


G = create_SBM_graph(sizes=[35,35,35],p_intra=[0.4,0.4,0.4],p_inter=0.001)
pos = draw_layout(G)

G = nx.random_geometric_graph(n=200, radius=0.2, seed=4)
pos = {n: d["pos"] for n, d in G.nodes(data=True)}

kind = 'tmp'


draw_graphs(G, pos)

# --- Precompute ---
nodes       = sorted(G.nodes())
dist_matrix = compute_distance_nodes(G)
quad_energy = precompute_energies(nodes, dist_matrix)

# --- Main loop ---
t0     = time.perf_counter()
scores = np.array([local_score(n, dist_matrix, quad_energy, k, temperature, lambda_reg) for n in nodes])
print(f"{k}-hop done in {time.perf_counter() - t0:.2f}s")

# --- Visualize ---
save_path = os.path.join('data','figures',kind,f"{k}_hop_{time.perf_counter() - t0:.2f}s.png")
os.makedirs(os.path.join('data','figures',kind),exist_ok=True)
draw_graph_with_values(G, pos, scores, title="Local Hyperbolicity Heatmap", save_path=save_path)
plot_hist(scores, title=f"Local Hyperbolicity (mean={scores.mean():.4f}, var={scores.var():.4f})", bins=20)