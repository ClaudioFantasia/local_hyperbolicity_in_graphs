from src.graphs.utils import * 
from src.hyperbolicity.gromov import *
from scipy.special import softmax

def max_gromov_entropic_distance_regularized(G, lambda_reg=0.5, T=0.1):
    dists = compute_distance_nodes(G)
    nodes_list = list(G.nodes())
    quadruplets = list(itertools.combinations(nodes_list, 4))

    scores = []
    deltas = []
    penalties = []

    for quad in quadruplets:
        # Calcolo Gromov
        delta = compute_delta_gromov(dists, quad)

        # regularizer on internal distances of nodes in the 4-tuple
        internal_dists = [dists[quad[i]][quad[j]] for i in range(4) for j in range(i+1, 4)]
        penalty = np.mean(internal_dists)
        
        deltas.append(delta)
        penalties.append(penalty)
    
    delta_arr = np.array(deltas)
    dist_arr  = np.array(penalties)


    den = delta_arr.max() - delta_arr.min()
    delta_norm = np.zeros_like(delta_arr) if den == 0 else (delta_arr - delta_arr.min()) / den
    dist_norm  = (dist_arr  - dist_arr.min())  / (dist_arr.max()  - dist_arr.min())

    # --- energy (maximize delta, minimize distance) ---
    scores = delta_norm - lambda_reg * dist_norm

    # --- softmax with temperature ---
    mu = softmax(scores / T)
 
    return mu, quadruplets

def min_gromov_entropic_distance_regularized(G, lambda_reg=0.5, T=0.1):
    dists = compute_distance_nodes(G)
    nodes_list = list(G.nodes())
    quadruplets = list(itertools.combinations(nodes_list, 4))

    scores = []
    deltas = []
    penalties = []

    for quad in quadruplets:
        # Calcolo Gromov
        delta = compute_delta_gromov(dists, quad)

        # regularizer on internal distances of nodes in the 4-tuple
        internal_dists = [dists[quad[i]][quad[j]] for i in range(4) for j in range(i+1, 4)]
        penalty = np.mean(internal_dists)
        
        deltas.append(delta)
        penalties.append(penalty)
    
    delta_arr = np.array(deltas)
    dist_arr  = np.array(penalties)

    den = delta_arr.max() - delta_arr.min()
    delta_norm = np.zeros_like(delta_arr) if den == 0 else (delta_arr - delta_arr.min()) / den
    dist_norm  = (dist_arr  - dist_arr.min())  / (dist_arr.max()  - dist_arr.min())

    # --- energy (maximize delta, minimize distance) ---
    scores =  -1 * delta_norm + lambda_reg * dist_norm

    # --- softmax with temperature ---
    mu = softmax(scores / T)
 
    return mu, quadruplets