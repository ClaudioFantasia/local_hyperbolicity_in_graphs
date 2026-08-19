import itertools
import random
from math import comb
import networkx as nx
def get_neighborhood(G, target, k, strategy='full_neighborhood', m=None, seed=None):
    """
    Node indices within k hops of `target`, i.e. Hop_k(target) in the
    notes' notation.

    `target` and the returned list are *positions* in list(G.nodes()) --
    the same index space as compute_distance_nodes / gromov_energy /
    dist_matrix -- not raw node ids. The two coincide for every graph
    currently used in the repo (nodes 0..n-1 in order), but the
    translation is done explicitly so a graph with different labels
    returns the right ball instead of silently returning a wrong one.

    Two strategies:
      - 'full_neighborhood': exact k-hop ball via BFS. Can blow up on
        hub nodes (common in citation graphs like Cora/CiteSeer), where
        a handful of hops already reaches most of the graph.
      - 'increasing_neighborhood': BFS that, at each hop, keeps at most
        `m` randomly sampled neighbors per already-visited node instead
        of all of them. This bounds the neighborhood size by roughly
        sum_{i=1}^{k} m^i regardless of how connected the graph is,
        which is what makes hub nodes tractable. `seed` controls which
        neighbors get kept.

    Returns a plain list of node indices (order is BFS order, not sorted).
    """
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    source = nodes[target]

    if strategy == 'full_neighborhood':
        lengths = nx.single_source_shortest_path_length(G, source, cutoff=k)
        return [index[n] for n in lengths]

    if strategy == 'increasing_neighborhood':
        rng = random.Random(seed)

        visited = {source}
        frontier = [source]
        result = [source]

        for _ in range(k):
            next_frontier = []
            seen_this_level = set()  # avoids duplicates across different frontier nodes

            for node in frontier:
                # candidates are based on `visited` as of the START of this hop
                candidates = [n for n in G.neighbors(node) if n not in visited]

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

        return [index[n] for n in result]

    raise ValueError(f"Unknown neighborhood strategy: {strategy!r}")


# ======================================================================
# 2. Sampling 4-tuples from a neighborhood (Sec 8, method 1)
# ======================================================================

def sampling_quads(neighborhood, MAX_SAMPLES=comb(100, 4), seed=None):
    """
    All 4-element combinations of `neighborhood`, or MAX_SAMPLES of them
    drawn uniformly at random if the exact count would exceed MAX_SAMPLES
    (comb(100, 4) ~= 3.9M). This caps both the O(n^4) blow-up in
    neighborhood size and the memory needed to hold every quad's score.

    Note that the cap makes the score of a large neighborhood an
    *estimate*, and a uniform one: on a 900-node ball it keeps 0.015% of
    the quads, while KL_score weights them by exp(-d_v(h)/T_geom). At
    small T_geom that mismatch matters -- see
    experiments/datasets/sampling_check.py.
    """
    n = len(neighborhood)
    num_quads = comb(n, 4)

    if num_quads <= MAX_SAMPLES:
        return list(itertools.combinations(neighborhood, 4))

    rng = random.Random(seed)
    quads = set()
    while len(quads) < MAX_SAMPLES:
        quads.add(tuple(sorted(rng.sample(neighborhood, 4))))
    return list(quads)
