import itertools
import random
from math import comb
import networkx as nx
def get_neighborhood(G, target, k, strategy='full_neighborhood', m=None, seed=None):
    """
    Node indices within k hops of `target`, i.e. Hop_k(target) in the
    notes' notation, either exactly or sampled.

    `target` and the returned list are *positions* in list(G.nodes()) --
    the same index space as compute_distance_nodes / gromov_energy /
    dist_matrix -- not raw node ids. The two coincide for every graph
    currently used in the repo (nodes 0..n-1 in order), but the
    translation is done explicitly so a graph with different labels
    returns the right ball instead of silently returning a wrong one.

    Four strategies. All of them return a *subset of the exact k-hop
    ball*, so k always means "nothing further away than k hops" whatever
    the strategy, and at fixed k they are directly comparable: they
    differ only in how much of that ball they keep.

      'full_neighborhood'       the exact ball, via BFS. Can blow up on
                                hub nodes (common in citation graphs
                                like Cora/CiteSeer), where a couple of
                                hops already reach most of the graph.
                                Uses k.
      'increasing_neighborhood' the GraphSAGE-style ball: at each hop
                                keep at most `m` randomly sampled new
                                neighbors per frontier node instead of
                                all of them. Bounds the size by roughly
                                sum_{i=1}^{k} m^i regardless of how
                                connected the graph is, which is what
                                makes hub nodes tractable. Uses k, m,
                                seed.
      'forest_fire'             Leskovec's forest fire: same shape, but
                                the number of neighbors burned per node
                                is *random* -- Geometric with mean m --
                                instead of fixed at m. Same expected
                                fanout as 'increasing_neighborhood', so
                                the pair isolates what the variance of
                                the fanout does on its own. The heavy
                                tail occasionally burns a whole
                                high-degree node, which is how forest
                                fire keeps the long paths a fixed
                                fanout cuts off.
                                Uses k, m, seed.
      'random_walk'             `m` independent random walks of k steps
                                from `target`, keeping every node
                                visited. Unlike the three above it does
                                not sweep hop by hop: it spends its
                                budget on a few long paths rather than
                                on a shell around the target, so it
                                reaches the k-th hop with far fewer
                                nodes -- and it revisits, so it favours
                                whatever the walk keeps coming back to
                                (high degree, and denser regions).
                                Uses k, m, seed.

    Only 'full_neighborhood' has "position in the result == distance
    from target". For the other three a node missed at hop i gets picked
    up at hop i+1 through another path, so the hop at which a node shows
    up is a sampling depth, an upper bound on its true distance and
    usually a loose one.

    Returns a plain list of node indices (order is visit order, not
    sorted; `target` is always first).
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

    if strategy == 'forest_fire':
        rng = random.Random(seed)

        visited = {source}
        frontier = [source]
        result = [source]

        for _ in range(k):
            next_frontier = []

            for node in frontier:
                candidates = [n for n in G.neighbors(node) if n not in visited]

                # quanti vicini bruciare: si continua a bruciarne uno in piu'
                # con probabilita' m/(m+1), cioe' una Geometrica di media
                # esattamente m -- lo stesso fanout atteso di
                # 'increasing_neighborhood', ma con una coda che ogni tanto
                # brucia un nodo intero
                burn = 0
                while burn < len(candidates) and rng.random() < m / (m + 1):
                    burn += 1

                # i bruciati si marcano subito: al contrario di
                # 'increasing_neighborhood' un fratello piu' avanti nella
                # frontiera non riproponera' gli stessi nodi
                for n in rng.sample(candidates, burn):
                    visited.add(n)
                    next_frontier.append(n)

            result.extend(next_frontier)
            frontier = next_frontier

            if not frontier:
                break

        return [index[n] for n in result]

    if strategy == 'random_walk':
        rng = random.Random(seed)

        visited = {source}
        result = [source]

        for _ in range(m):
            node = source
            for _ in range(k):
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    break            # nodo isolato: il cammino non parte
                node = rng.choice(neighbors)
                if node not in visited:
                    visited.add(node)
                    result.append(node)

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
