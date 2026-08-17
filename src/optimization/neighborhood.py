import itertools
import random
from math import comb
import networkx as nx 
def get_neighborhood(G, target, k, strategy='full_neighborhood', m=None, seed=None):
    """
    Node indices within k hops of `target`, i.e. Hop_k(target) in the
    notes' notation.

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

        return result

    raise ValueError(f"Unknown neighborhood strategy: {strategy!r}")


# ======================================================================
# 2. Sampling 4-tuples from a neighborhood (Sec 8, method 1)
# ======================================================================

def sampling_quads(neighborhood, MAX_SAMPLES=comb(100,4)):
    """
    All 4-element combinations of `neighborhood`, or MAX_SAMPLES of them
    drawn uniformly at random if the exact count would exceed MAX_SAMPLES
    (comb(100, 4) ~= 3.9M). This caps both the O(n^4) blow-up in
    neighborhood size and the memory needed to hold every quad's score.
    """
    n = len(neighborhood)
    num_quads = comb(n, 4)

    if num_quads <= MAX_SAMPLES:
        return list(itertools.combinations(neighborhood, 4))

    quads = set()
    while len(quads) < MAX_SAMPLES:
        quads.add(tuple(sorted(random.sample(neighborhood, 4))))
    return list(quads)
