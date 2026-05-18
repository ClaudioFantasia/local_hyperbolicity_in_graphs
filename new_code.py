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
import yaml 

# ==========================================
# Defining graph structure
# ==========================================
G = nx.random_geometric_graph(n=50, radius=0.35, dim=2, seed=42)
pos = {node[0]: node[1]['pos'] for node in G.nodes(data=True)}

def get_opt_pm():
    with open("configs/optimization_parameters.yaml", "r") as file:
        config = yaml.safe_load(file)
    config = config['optimization']
    return config['temperature'],config['alpha_diffusion'],config['time'],config['localized_nodes'],config['lambda_reg']
temperature,alpha_diffusion,time,localized_nodes,lambda_reg = get_opt_pm()
print(f"The parameters that I am using are {get_opt_pm()}")
G = nx.cycle_graph(n=50)
pos = draw_layout(G)
#draw_graphs(G, pos)
#G.remove_edge(48,49)


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
W_local = local_weights_distance_based(quads, localized_nodes, dist_matrix)
E_gromov = normalize(E_gromov)
W_local = normalize(W_local)
D_energy = distance_energy(quads, dist_matrix) 
"""
W_local = np.zeros(len(quads))
for quad in quads:
    if any(x in quad for x in [16]):
        W_local[list(quad)] = 1
W_local /= np.sum(W_local)
"""

print(f"Mean local weights: {np.mean(W_local)}")
print(f"Var  local weights: {np.var(W_local)}")

# ==========================================
# 2. COMPOSIZIONE DELL'OBIETTIVO
# ==========================================
cost_vector = E_gromov + lambda_reg * W_local

# ==========================================
# 3. SOLUZIONE E VALUTAZIONE
# ==========================================
mu = solve_entropic_regularization(cost_vector, temperature)

print(f"Per curiosita la prima parte e: {np.sum(mu *cost_vector)}")
print(f"la seconda parte e: {temperature * np.sum(mu * np.log(mu))}")

cost_function_val = np.sum(mu * cost_vector  - temperature * mu * np.log(mu))
print(f"La cost function e': {cost_function_val}")

plot_hist(W_local, title=f"Local weights (Mean={np.mean(W_local):.4f}, Var={np.var(W_local):.4f})", bins=50)
plot_hist(mu, title="$\\mu$ distribution (Entropic Solution)")

