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

def score_max(target, dist_matrix, quad_cache, k):
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quads = list(itertools.combinations(neighborhood, 4))
    if not quads:
        return 0.0

     # Calcola in batch solo le tuple non ancora in cache
    missing = [q for q in quads if q not in quad_cache]
    if missing:
        energies = gromov_energy(missing, dist_matrix)
        quad_cache.update(zip(missing, energies))

    e_gromov = np.array([quad_cache[tuple(sorted(q))] for q in quads])
    return float(np.max(e_gromov))


def score_softmax(target, dist_matrix, quad_energy, k, temperature):
    neighborhood = np.where((dist_matrix[target] <= k) & (dist_matrix[target] > 0))[0]
    quads = [(target, x, y, z) for x, y, z in itertools.combinations(neighborhood, 3)]
    if not quads:
        return 0.0

    e_gromov = np.array([quad_energy[tuple(sorted(q))] for q in quads])
    dists    = dist_matrix[target, np.array(list(q[1:] for q in quads))]
    weights  = softmax(-np.sum(dists, axis=1) / temperature)
    return float(np.sum(e_gromov * weights))


def score_KL_divergence(target, dist_matrix, quad_cache, k, temperature, geometry_temperature = 1):
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quads = list(itertools.combinations(neighborhood, 4))
    if not quads:
        return 0.0
    
    # Calcola in batch solo le tuple non ancora in cache
    missing = [q for q in quads if q not in quad_cache]
    if missing:
        energies = gromov_energy(missing, dist_matrix)
        quad_cache.update(zip(missing, energies))

    w_local = np.mean(dist_matrix[np.array(quads), target], axis=1)
    w_local = softmax(-1 * w_local / geometry_temperature)

    e_gromov = np.array([quad_cache[q] for q in quads])

    mu = solve_KL_regularization(e_gromov, w_local, temperature)

    return float(np.sum(mu * e_gromov - KL_divergence(mu, w_local, temperature)))


def score_entropic(target, dist_matrix, quad_cache, k, temperature, lambda_reg):
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quads = list(itertools.combinations(neighborhood, 4))
    if not quads:   
        return 0.0

    # Calcola in batch solo le tuple non ancora in cache
    missing = [q for q in quads if q not in quad_cache]
    if missing:
        energies = gromov_energy(missing, dist_matrix)
        quad_cache.update(zip(missing, energies))

    e_gromov = normalize(np.array([quad_cache[q] for q in quads]))
    w_local  = normalize(np.mean(dist_matrix[np.array(quads), target], axis=1))

    cost = e_gromov - lambda_reg * w_local
    mu   = solve_entropic_regularization(cost, temperature)
    return float(np.sum(mu * cost - temperature * mu * np.log(mu)))
