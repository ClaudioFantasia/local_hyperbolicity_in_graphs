import numpy as np
import itertools
from scipy.special import softmax, logsumexp
from src.optimization.objectives import gromov_energy, compute_distance_nodes
import networkx as nx
import random 
from math import comb

SAMPLE_THRESHOLD = 100
MAX_SAMPLES = comb(100,4)

def sampling_quads(neighborhood):
    n = len(neighborhood)
    num_quads = comb(n, 4)
    
    if num_quads <= MAX_SAMPLES:
        quads = list(itertools.combinations(neighborhood, 4))
    else:
        quads = set()
        while len(quads) < MAX_SAMPLES:
            quads.add(tuple(sorted(random.sample(neighborhood, 4))))
        quads = list(quads)
    return quads

def get_neighborhood(G, target, k, strategy='full_neighborhood', m=None, seed=None):
    """
    It returns the list of node_labels that are in the k-hop of target node in the graph.
    """
    if strategy == 'full_neighborhood':
        lengths = nx.single_source_shortest_path_length(G, target, cutoff=k)
        return list(lengths.keys())

    if strategy == 'increasing_neighborhood':
        rng = random.Random(seed)

        visited = {target}
        frontier = [target]
        result = [target]

        for _ in range(k):
            next_frontier = []
            seen_this_level = set()  # per evitare duplicati tra nodi diversi dello stesso frontier

            for node in frontier:
                neighbors = list(G.neighbors(node))
                # candidati basati sullo stato di visited ALL'INIZIO del livello
                candidates = [n for n in neighbors if n not in visited]

                if m is not None and len(candidates) > m:
                    sampled = rng.sample(candidates, m)
                else:
                    sampled = candidates

                for n in sampled:
                    if n not in seen_this_level:
                        seen_this_level.add(n)
                        next_frontier.append(n)

            visited.update(seen_this_level)
            result.extend(next_frontier)
            frontier = next_frontier

            if not frontier:
                break

        return result


def KL_score(G, target, quad_cache, k, temperature, geometric_temperature, dist_matrix, strategy='full_neighborhood', m=5):
    neighborhood = get_neighborhood(G, target, k, strategy=strategy, m=m)

    return len(neighborhood)
    print(f"Just curios: how many neighbors -- {len(neighborhood)} and target node {target}")
    if len(neighborhood) < 4:       
        return np.zeros_like(np.atleast_1d(geometric_temperature), dtype=float)
  

    geometric_temperature = np.atleast_1d(geometric_temperature)

    quads = sampling_quads(neighborhood)

    q_arr = np.array(quads, dtype=np.int64)

    w_local = dist_matrix[q_arr, target].mean(axis=1)

    keys = [tuple(sorted(q)) for q in quads]

    missing = [key for key in keys if key not in quad_cache]

    if missing:
        quad_cache.update(zip(missing, gromov_energy(missing, dist_matrix)))

    e_gromov = np.array([quad_cache[k] for k in keys])

    # V* = beta * (log sum_i exp(-w_i/T_geom + delta_i/beta) - log sum_j exp(-w_j/T_geom))
    log_numerator   = logsumexp(-w_local[:, None] / geometric_temperature[None, :] 
                                + e_gromov[:, None] / temperature, axis=0)
    log_denominator = logsumexp(-w_local[:, None] / geometric_temperature[None, :], axis=0)

    return temperature * (log_numerator - log_denominator)



def old_get_neighborhood(G, target, k):
    lengths = nx.single_source_shortest_path_length(G, target, cutoff=k)
    neighborhood_nodes = list(lengths.keys())
    subG = G.subgraph(neighborhood_nodes)
    dist_matrix, node_index = compute_distance_nodes(subG)
    return neighborhood_nodes, dist_matrix, node_index




def batched_subgraph_score_KL_divergence(G, target, quad_cache, k, temperature, geometric_temperature, batch_size=1_000_000):
    neighborhood, dist_matrix, node_index = _get_neighborhood(G, target, k)

    if len(neighborhood) < 4:
        return np.zeros_like(np.atleast_1d(geometric_temperature), dtype=float)

    geometric_temperature = np.atleast_1d(geometric_temperature)
    target_idx = node_index[target]
    quad_iter = itertools.combinations(neighborhood, 4)
    gammaexp_summation = 0

    while True:
        quads = list(itertools.islice(quad_iter, batch_size))
        if not quads:
            break

        ## DISTANCE DISTRIBUTION
        q_idx = np.array([[node_index[n] for n in q] for q in quads])
        w_local = dist_matrix[q_idx, target_idx].mean(axis=1)
        gamma = softmax(-w_local[:, None] / geometric_temperature[None, :], axis=0)

        ## GROMOV COMPUTATION
        keys = [tuple(sorted(q)) for q in quads]
        missing = [key for key in keys if key not in quad_cache]
        if missing:
            missing_idx = [[node_index[n] for n in q] for q in missing]
            quad_cache.update(zip(missing, gromov_energy(missing_idx, dist_matrix)))
        e_gromov = np.array([quad_cache[k] for k in keys])[:, None]


        gammaexp_summation += (gamma * np.exp(e_gromov / temperature)).sum(axis=0)

    return temperature * np.log(gammaexp_summation)



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