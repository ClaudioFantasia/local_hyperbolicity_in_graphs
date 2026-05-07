from src.hyperbolicity.gromov import *
from scipy.linalg import expm
import numpy as np
from scipy.sparse.linalg import factorized

def gromov_energy(quads, dist_matrix):
    E = np.zeros(shape = (len(quads)))
    for i, quad in enumerate(quads):
        E[i] = compute_delta_gromov(dist_matrix,quad)
    return E

def distance_energy(quads, dist_matrix):
    P = np.zeros(shape = (len(quads)))
    for i, quad in enumerate(quads):
        P[i] = compute_intra_distance(dist_matrix, quad)
    return P 

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
    weights = weights / np.sum(weights)
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
    return temperature * x * np.log(x / (y + eps))
