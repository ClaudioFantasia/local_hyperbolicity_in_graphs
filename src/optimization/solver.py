from src.graphs.utils import * 
from src.hyperbolicity.gromov import *
from scipy.special import softmax
from src.optimization.spectral_regularization import *
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



def max_gromov_spectral_regularized(G, lambda_reg=0.5, T=0.1, alpha = 0.5):
    dists = compute_distance_nodes(G)
    nodes_list = list(G.nodes())
    quadruplets = list(itertools.combinations(nodes_list, 4))

    scores = []
    deltas = []
    penalties = []

    for quad in quadruplets:
        # Calcolo Gromov
        delta = compute_delta_gromov(dists, quad)

        deltas.append(delta)
    
    penalties = list(compute_energy(G,alpha).values())

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


def max_gromov_l2_regularization(G, x0, lambda_reg = 0.5, max_steps = 10):
    # lets suppose x0 is the uniform distribution as starting point (it must be in the simplex)
    nodes_list = list(G.nodes())
    quadruplets = list(itertools.combinations(nodes_list, 4))
    dists = compute_distance_nodes(G)

    deltas = []
    scores = []

    N = len(quadruplets)
    u = np.full(shape = N, fill_value = 1/N) # uniform distribution

    for quad in quadruplets:
        # Calcolo Gromov
        delta = compute_delta_gromov(dists, quad)

        deltas.append(delta)
    deltas = np.array(deltas)
    xk = x0
    step_size = 0.25 # Lip should be 2, so step = 0.5

    for k in range(0,max_steps):
        grad_fk = deltas - 2 * lambda_reg * (xk - u)
        xk = xk + step_size * grad_fk
        xk = project_to_simplex(xk)
    return xk, quadruplets



def project_to_simplex(y):
    """
    you want to find x close to y that satisfy the simplex constraints
    """
    y = np.asarray(y)
    n = y.shape[0]

    # Step 1: sort in descending order
    u = np.sort(y)[::-1]

    # Step 2: cumulative sum
    cssv = np.cumsum(u)

    # Step 3: find rho
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    rho = rho[-1]

    # Step 4: compute tau
    tau = (cssv[rho] - 1) / (rho + 1)

    # Step 5: projection
    x = np.maximum(y - tau, 0)

    return x





def max_gromov_entropic_regularization_iterative(G, lambda_reg=0.5, T=0.1, max_steps=20):
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

    step_size = 0.25

    for k in range(0,max_steps):
        grad_fk = deltas - (np.log(xk) + 1)
        xk = xk + step_size * grad_fk
        xk = project_to_simplex(xk)
    return xk, quadruplets
