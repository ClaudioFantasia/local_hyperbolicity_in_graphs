import itertools
import random
from math import comb

import networkx as nx
import numpy as np
from scipy.special import logsumexp, softmax

from src.optimization.objectives import gromov_energy, tuple_distance
from src.optimization.neighborhood import sampling_quads, get_neighborhood

def KL_score(G, target, quad_cache, k, temperature, geometric_temperature,
             dist_matrix, strategy='full_neighborhood', m=5, seed=None):
    """
    Local Gromov hyperbolicity of `target`, under the KL-divergence
    formulation

    Conceptually: build a reference distribution rho_v over 4-tuples
    that favours quads close to `target` (governed by
    geometric_temperature), then solve

        max_mu  sum_i mu_i * delta(h_i)  -  temperature * KL(mu || rho_v)

    whose closed-form optimal value (Sec 5) is

        V* = temperature * ( logsumexp_i[ -w_i/T_geom + delta_i/temperature ]
                            - logsumexp_j[ -w_j/T_geom ] )

    where w_i = d_v(h_i) is the mean distance from quad h_i to `target`,
    and delta_i = gromov_energy(h_i). The two logsumexp terms are
    exactly log(unnormalised numerator) and log(rho_v's normalising
    constant); computing V* this way (rather than materialising rho_v
    and mu explicitly) avoids overflow/underflow in the exponentials.

    As temperature -> 0, V* converges to delta(h) of the single quad
    that maximises Gromov energy near `target` (the "sharpest" local
    estimate); as temperature -> inf, V* approaches the rho_v-weighted
    average Gromov energy over the whole neighborhood (smoothest).

    Parameters
    ----------
    G : networkx.Graph
    target : node index (into dist_matrix / quad_cache's index space)
    quad_cache : dict, tuple(sorted(quad)) -> gromov_energy(quad).
        Mutated in place so repeated calls (e.g. one per node) reuse
        Gromov energies already computed for shared quads.
    k : neighborhood radius (hops)
    temperature : regularisation strength; see V* formula above
    geometric_temperature : scalar or array. Controls how quickly rho_v
        decays with distance from `target`; can be swept as an array to
        get one score per value without recomputing quads/energies.
    dist_matrix : all-pairs distance matrix (see compute_distance_nodes)
    strategy, m : passed through to get_neighborhood
    seed : controls both random choices behind the score -- which
        neighbors 'increasing_neighborhood' keeps, and which quads
        sampling_quads draws once the exact count exceeds its cap.
        None (the default) leaves both unseeded, as before.

    Returns
    -------
    np.ndarray, shape matching np.atleast_1d(geometric_temperature):
    one G* value per geometric_temperature.
    """
    neighborhood = get_neighborhood(G, target, k, strategy=strategy, m=m, seed=seed)

    if len(neighborhood) < 4:
        # can't form a single 4-tuple -- no local hyperbolicity to report
        return np.zeros_like(np.atleast_1d(geometric_temperature), dtype=float)

    geometric_temperature = np.atleast_1d(geometric_temperature)
    quads = sampling_quads(neighborhood, seed=seed)

    # w_local[i] = d_v(h_i): mean distance from quad i's four nodes to target
    w_local = tuple_distance(quads, dist_matrix, target)

    # Gromov energies, memoised across calls via quad_cache
    keys = [tuple(sorted(q)) for q in quads]
    missing = [key for key in keys if key not in quad_cache]
    if missing:
        quad_cache.update(zip(missing, gromov_energy(missing, dist_matrix)))
    e_gromov = np.array([quad_cache[key] for key in keys])

    # G*(target) = temperature * (logsumexp[-w/T_geom + delta/temperature] - logsumexp[-w/T_geom])
    log_numerator = logsumexp(
        -w_local[:, None] / geometric_temperature[None, :]
        + e_gromov[:, None] / temperature,
        axis=0,
    )
    log_denominator = logsumexp(
        -w_local[:, None] / geometric_temperature[None, :],
        axis=0,
    )

    return temperature * (log_numerator - log_denominator)

