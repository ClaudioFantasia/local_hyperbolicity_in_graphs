import numpy as np
import itertools
from scipy.special import softmax, logsumexp



def gromov_energy(quads, dist_matrix):
    q = np.array(quads)          # (N, 4)
    x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]

    s0 = dist_matrix[x, y] + dist_matrix[z, w]
    s1 = dist_matrix[x, z] + dist_matrix[y, w]
    s2 = dist_matrix[x, w] + dist_matrix[y, z]

    top2 = np.sort(np.stack([s0, s1, s2], axis=1), axis=1)  # (N, 3)
    return (top2[:, 2] - top2[:, 1]) / 2.0


def _get_quads_and_energies(target, dist_matrix, quad_cache, k):
    """Helper to find neighborhood quads and batch-compute missing energies."""
    neighborhood = np.where(dist_matrix[target] <= k)[0]
    print(len(neighborhood))
    quads = list(itertools.combinations(neighborhood, 4))
    if not quads:
        return quads, np.array([])

    missing = [q for q in quads if q not in quad_cache]
    if missing:
        energies = gromov_energy(missing, dist_matrix)
        quad_cache.update(zip(missing, energies))

    e_gromov = np.array([quad_cache[tuple(sorted(q))] for q in quads])
    return quads, e_gromov

def score_KL_divergence(target, dist_matrix, quad_cache, k, temperature, geometric_temperature):
    quads, e_gromov = _get_quads_and_energies(target, dist_matrix, quad_cache, k)
    if not quads:
        return np.zeros_like(np.atleast_1d(geometric_temperature), dtype=float)
    
    geometric_temperature = np.atleast_1d(geometric_temperature)  # shape (T,)
    
    w_local = np.mean(dist_matrix[np.array(quads), target], axis=1)  # shape (N,)
    
    # gamma: (N, T) -- one gamma column per temperature, normalized along N (axis=0)
    gamma = softmax(-1 * w_local[:, None] / geometric_temperature[None, :], axis=0)
    
    # e_gromov broadcast to (N, T)
    e_gromov = np.tile(np.asarray(e_gromov).reshape(-1, 1), (1, geometric_temperature.shape[0]))
    
    # V*: shape (T,) -- one value per geometric_temperature
    # V* = beta * log sum_j gamma(h_j) * exp(delta(h_j) / beta)
    log_terms = np.log(gamma) + e_gromov / temperature  # log-sum-exp trick
    V_star = temperature * logsumexp(log_terms, axis=0)  # shape (T,)
    
    return V_star

def score_KL_divergence_batched(target, dist_matrix, quad_cache, k, temperature,
                                  geometric_temperature, batch_size=5_000_000):
    geometric_temperature = np.atleast_1d(geometric_temperature)
    T = geometric_temperature.shape[0]

    neighborhood = np.where(dist_matrix[target] <= k)[0]
    quad_iter = itertools.combinations(neighborhood, 4)


    m_num, s_num = np.full(T, -np.inf), np.zeros(T)   # numerator LSE state
    m_den, s_den = np.full(T, -np.inf), np.zeros(T)   # denominator LSE state

    total_quads = 0
    while True:
        if len(quad_cache) > 10_000_005:
            quad_cache = {}
        batch = list(itertools.islice(quad_iter, batch_size))
        if not batch:
            break
        total_quads += len(batch)

        missing = [q for q in batch if q not in quad_cache]
        if missing:
            e_missing = gromov_energy(missing, dist_matrix)
            quad_cache.update(zip(missing, e_missing))
        e_gromov = np.array([quad_cache[tuple(sorted(q))] for q in batch])

        w_local = np.mean(dist_matrix[np.array(batch), target], axis=1)

        base = -w_local[:, None] / geometric_temperature[None, :]
        num_terms = base + e_gromov[:, None] / temperature

        # update numerator accumulator
        batch_max = np.max(num_terms, axis=0)
        new_m = np.maximum(m_num, batch_max)
        s_num = s_num * np.exp(m_num - new_m) + np.sum(np.exp(num_terms - new_m), axis=0)
        m_num = new_m

        # update denominator accumulator
        batch_max = np.max(base, axis=0)
        new_m = np.maximum(m_den, batch_max)
        s_den = s_den * np.exp(m_den - new_m) + np.sum(np.exp(base - new_m), axis=0)
        m_den = new_m

    if total_quads == 0:
        return np.zeros_like(np.atleast_1d(geometric_temperature), dtype=float)

    lse_num = m_num + np.log(s_num)
    lse_den = m_den + np.log(s_den)
    return temperature * (lse_num - lse_den)    













###
"""
I am completely lost, I am just gonna rewrite the important module above
"""
###








def precompute_energies(nodes, dist_matrix):
    quads = list(itertools.combinations(nodes, 4))
    energies = gromov_energy(quads, dist_matrix)
    print(f"Global Gromov hyperbolicity: {np.max(energies):.4f}")
    return dict(zip(quads, energies))



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





def score_entropic(target, dist_matrix, quad_cache, k, temperature, lambda_reg):
    quads, e_gromov = _get_quads_and_energies(target, dist_matrix, quad_cache, k)
    if not quads:   
        return 0.0

    w_local  = np.mean(dist_matrix[np.array(quads), target], axis=1)

    cost = e_gromov - lambda_reg * w_local
    mu   = solve_entropic_regularization(cost, temperature)
    return float(np.sum(mu * cost - temperature * mu * np.log(mu)))

def score_softing(target, dist_matrix, quad_cache, k, sigma):
    quads, e_gromov = _get_quads_and_energies(target, dist_matrix, quad_cache, k)
    if not quads:   
        return 0.0

    max_dist = np.max(dist_matrix[np.array(quads), target], axis=1)  # (n_quads,)
    weights = np.exp(-max_dist / sigma)
    weighted_delta = weights * e_gromov

    return float(np.max(weighted_delta))