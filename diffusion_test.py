from src.optimization.objectives import *
from src.graphs.visualization import *
G = nx.random_geometric_graph(n=50, radius=0.35, dim=2, seed=42)
pos = {node[0]: node[1]['pos'] for node in G.nodes(data=True)}

localized_nodes = [17]

alpha_diffusion = 5
time = 5

nodes_list = list(G.nodes())
quads = list(itertools.combinations(nodes_list, 4))
L = nx.normalized_laplacian_matrix(G).toarray()
dist_matrix = compute_distance_nodes(G)

results = diffuse_signal_heat_kernel([localized_nodes], L, time)
for x in results:
    draw_graph_with_values(G,pos,x)