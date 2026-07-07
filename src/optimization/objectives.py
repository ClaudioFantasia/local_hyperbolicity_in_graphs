
from scipy.linalg import expm
import numpy as np
from scipy.sparse.linalg import factorized
from scipy.special import softmax
import networkx as nx

###########################################
###         GROMOV
###########################################

def gromov_energy(quads, dist_matrix):
    q = np.array(quads)
    x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]
    s0 = dist_matrix[x, y] + dist_matrix[z, w]
    s1 = dist_matrix[x, z] + dist_matrix[y, w]
    s2 = dist_matrix[x, w] + dist_matrix[y, z]
    top2 = np.sort(np.stack([s0, s1, s2], axis=1), axis=1)
    return (top2[:, 2] - top2[:, 1]) / 2.0

def compute_gromov_hyperbolicity(G):
    """
    Compute Gromov hyperbolicity from a graph.

    N.B. Be careful that we are allocating the full vector of quads. This code can run on maximum 300 nodes.
    """   
    dist_matrix = compute_distance_nodes(G)
    nodes = list(G.nodes())
    deltas = {}

    quads = list(itertools.combinations(nodes,4))

    deltas = gromov_energy(quads,dist_matrix)
    
    max_delta = np.max(deltas)
    return max_delta

###########################################
###         UTILS
###########################################

def normalize(x):
    den = np.max(x) - np.min(x)
    if den != 0:
        x = (x - np.min(x)) / den 
    else: 
        x = np.zeros(shape=np.shape(x))
    return x  

def negEntropy_regularization(x, temperature):
    return temperature * x * np.log(x)

def l2_regularization(x, lambda_reg):
    return lambda_reg * 0.5 * (x**2)

def KL_divergence(x, y, temperature, eps = 1e-16):
    #x_safe = np.clip(x, eps, None)
    return temperature * x * np.log(x / (y + eps))

def compute_distance_nodes(G):
    """
    Compute the distances between nodes using shortest path as metric.
    It return both the matrix and the index dictionary.
    """
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    dist = np.full((n, n), np.inf)

    for u, lengths in nx.all_pairs_shortest_path_length(G):
        i = index[u]
        for v, d in lengths.items():
            dist[i, index[v]] = d

    return dist, index











def distance_energy(quads, dist_matrix):
    P = np.zeros(shape = (len(quads)))
    for i, quad in enumerate(quads):
        P[i] = compute_intra_distance(dist_matrix, quad)
    return P 

def exp_distance_energy(distance_energy, geometry_temperature = 1):
    return softmax(-distance_energy / geometry_temperature)


def spectral_energy(quads, L, alpha_diffusion):
    E = np.zeros(len(quads))
    results = diffuse_signal_denoising_solve(quads, L, alpha_diffusion)
    for i, x in enumerate(results): 
        E[i] = x.T @ L @ x   
    return E

def local_weights_diffuse_based_alpha(quads, localized_nodes, L, alpha_diffusion=0.5):
    weights = np.zeros(len(quads))
    results = diffuse_signal_denoising_solve([localized_nodes],L,alpha_diffusion)
    x = results[0]
    for cnt, (i,j,k,l) in enumerate(quads):
        weights[cnt] = np.mean([x[i],x[j],x[k],x[l]])
    weights = weights / np.sum(weights)
    return weights

def local_weights_diffuse_based_time(quads, localized_nodes, L, time=1):
    weights = np.zeros(len(quads))
    results = diffuse_signal_heat_kernel([localized_nodes],L,time)
    x = results[0]
    for cnt, (i,j,k,l) in enumerate(quads):
        weights[cnt] = np.mean([x[i],x[j],x[k],x[l]])
    weights = weights / np.sum(weights)
    return weights

def local_weights_distance_based(quads, localized_nodes, dist_matrix):
    weights = np.zeros(len(quads))
    for cnt, quad in enumerate(quads):
        weights[cnt] = np.mean(dist_matrix[np.ix_(quad,localized_nodes)])
    #weights = weights / np.sum(weights)
    return weights

def diffuse_signal_denoising_solve(quads, L, alpha):
    """
    nodes: list of nodes, or list of nodes list
    L : laplacian
    alpha : parameter for diffusion
    """
    n = L.shape[0]
    M = np.eye(n) + alpha * L
    solve = factorized(M)  
    results = []
    for quad in quads:
        y = np.zeros(n)
        quad = [q for q in quad]
        y[quad] = 1
        x = solve(y)
        results.append(x)
    return results

def diffuse_signal_heat_kernel(quads, L, time):
    n = L.shape[0]
    results = []
    for quad in quads:
        y = np.zeros(n)
        quad = [q for q in quad]
        y[quad] = 1
        x = expm(-time * L) @ y
        results.append(x)
    return results


