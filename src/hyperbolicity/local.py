import itertools
import numpy as np
from scipy.special import softmax

from src.optimization.objectives import gromov_energy, normalize
from src.optimization.solver import solve_entropic_regularization, solve_KL_regularization
from src.optimization.objectives import distance_energy, exp_distance_energy, KL_divergence
from scipy.special import softmax

def precompute_energies(nodes, dist_matrix):
    quads = list(itertools.combinations(nodes, 4))
    energies = gromov_energy(quads, dist_matrix)
    print(f"Global Gromov hyperbolicity: {np.max(energies):.4f}")
    return dict(zip(quads, energies))

def _get_quads_and_energies(target, dist_matrix, quad_cache, k):
    """Helper to find neighborhood quads and batch-compute missing energies."""
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quads = list(itertools.combinations(neighborhood, 4))
    if not quads:
        return quads, np.array([])

    missing = [q for q in quads if q not in quad_cache]
    if missing:
        energies = gromov_energy(missing, dist_matrix)
        quad_cache.update(zip(missing, energies))

    e_gromov = np.array([quad_cache[tuple(sorted(q))] for q in quads])
    return quads, e_gromov

def score_max(target, dist_matrix, quad_cache, k):
    quads, e_gromov = _get_quads_and_energies(target, dist_matrix, quad_cache, k)
    if not quads:
        return 0.0

    return float(np.max(e_gromov))


def score_softmax(target, dist_matrix, quad_cache, k, temperature):
    neighborhood = np.where((dist_matrix[target] <= k) & (dist_matrix[target] > 0))[0]
    quads = [tuple(sorted((target, x, y, z))) for x, y, z in itertools.combinations(neighborhood, 3)]
    if not quads:
        return 0.0

    missing = [q for q in quads if q not in quad_cache]
    if missing:
        energies = gromov_energy(missing, dist_matrix)
        quad_cache.update(zip(missing, energies))

    e_gromov = np.array([quad_cache[q] for q in quads])
    dists    = dist_matrix[target, np.array(list(q for q in quads))[:, 1:]] # Just a simplification, actually need the elements that are not target. 
    # But wait, earlier it was list(q[1:] ... ), which assumed the first element was the target. Since we sorted them, it's safer to just do set subtraction, or we can use the original tuple for distance.

    # Let's keep the original tuple logic for dists
    original_quads = [(target, x, y, z) for x, y, z in itertools.combinations(neighborhood, 3)]
    dists = dist_matrix[target, np.array(list(q[1:] for q in original_quads))]
    weights = softmax(-np.sum(dists, axis=1) / temperature)
    return float(np.sum(e_gromov * weights))


def score_KL_divergence(target, dist_matrix, quad_cache, k, temperature, geometric_temperature):
    quads, e_gromov = _get_quads_and_energies(target, dist_matrix, quad_cache, k)
    if not quads:
        return 0.0
    
    w_local = np.mean(dist_matrix[np.array(quads), target], axis=1)
    w_local = softmax(-1 * w_local / geometric_temperature)

    mu = solve_KL_regularization(e_gromov, w_local, temperature)
    return float(np.sum(mu * e_gromov - KL_divergence(mu, w_local, temperature)))


def score_entropic(target, dist_matrix, quad_cache, k, temperature, lambda_reg):
    quads, e_gromov = _get_quads_and_energies(target, dist_matrix, quad_cache, k)
    if not quads:   
        return 0.0

    e_gromov = normalize(e_gromov)
    w_local  = normalize(np.mean(dist_matrix[np.array(quads), target], axis=1))

    cost = e_gromov - lambda_reg * w_local
    mu   = solve_entropic_regularization(cost, temperature)
    return float(np.sum(mu * cost - temperature * mu * np.log(mu)))
