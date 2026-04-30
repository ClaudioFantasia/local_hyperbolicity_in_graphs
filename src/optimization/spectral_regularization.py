import networkx as nx 
import numpy as np 
import itertools

def compute_spectral_distance(N, quad, L, H, alpha):

    y = np.zeros(N)
    quad = [q for q in quad]
    y[quad] = 1
    x = H @ y
    return x.T @ L @ x

def compute_energy(G, alpha):
    N = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G).toarray()
    energies = {}
    H = np.linalg.inv(np.eye(N) + alpha *L) 
    nodes_list = list(G.nodes())
    quadruplets = list(itertools.combinations(nodes_list, 4))
    for quad in quadruplets:
        energy = compute_spectral_distance(N,quad,L,H,alpha)
        energies[quad] = energy
    return energies
