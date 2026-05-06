from src.optimization.solver import *
from src.optimization.objectives import *
import numpy as np
import networkx as nx
import itertools
import matplotlib.pyplot as plt
from src.graphs.visualization import *

from src.graphs.utils import compute_distance_nodes
from src.graphs.visualization import plot_hist
from src.hyperbolicity.gromov import compute_gromov_hyperbolicity

G = nx.random_geometric_graph(n=50, radius=0.35, dim=2, seed=42)
pos = {node[0]: node[1]['pos'] for node in G.nodes(data=True)}

G = nx.cycle_graph(n=50)
pos = draw_layout(G)
draw_graphs(G, pos)
G.remove_edge(48,49)
localized_nodes = [17]
alpha_diffusion = 5
temperature = 0.1
lambda_reg = 10
time = 5

nodes_list = list(G.nodes())
quads = list(itertools.combinations(nodes_list, 4))
L = nx.normalized_laplacian_matrix(G).toarray()
dist_matrix = compute_distance_nodes(G)

max_delta, _ = compute_gromov_hyperbolicity(G)
print(f"Global gromov hyperbolicity: {max_delta}")

# ==========================================
# 1. COSTRUZIONE DEI PEZZETTINI MATEMATICI
# ==========================================
E_gromov = gromov_energy(quads, dist_matrix)
#W_local  = local_weights_distance_based(quads, localized_nodes, dist_matrix)
W_local = local_weights_diffuse_based_alpha(quads, localized_nodes, L, alpha_diffusion)
W_local = np.zeros(len(quads))
for quad in quads:
    if any(x in quad for x in [16]):
        W_local[list(quad)] = 1
W_local /= np.sum(W_local)


print(f"Mean local weights: {np.mean(W_local)}")
print(f"Var  local weights: {np.var(W_local)}")

# ==========================================
# 2. COMPOSIZIONE DELL'OBIETTIVO
# ==========================================
cost_vector = E_gromov

# ==========================================
# 3. SOLUZIONE E VALUTAZIONE
# ==========================================
mu = solve_KL_regularization(cost_vector, W_local, temperature)
print(f"Per curiosita la prima parte e: {np.sum(mu *cost_vector)}")
print(f"la seconda parte e: {np.sum(temperature * mu * np.log(mu / (W_local + 1e-16)))}")

cost_function_val = np.sum(mu * cost_vector  - temperature * mu * np.log(mu / (W_local + 1e-16)))
print(f"La cost function e': {cost_function_val}")

plot_hist(W_local, title=f"Local weights (Mean={np.mean(W_local):.4f}, Var={np.var(W_local):.4f})", bins=50)
plot_hist(mu, title="$\\mu$ distribution (Entropic Solution)")

