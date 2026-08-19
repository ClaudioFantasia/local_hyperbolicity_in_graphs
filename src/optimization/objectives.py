"""
Exact Gromov hyperbolicity, and the building blocks (distances,
regularisers, locality weights) that the rest of src/optimization
combines into the *local* hyperbolicity score in local.py.

Notation follows Research_notes_Local_Hyperbolicity_in_Graphs.pdf:
  - A "quad" h = (x, y, z, w) is a 4-tuple of node *indices*, i.e.
    positions in sorted(G.nodes()) / list(G.nodes()) -- not raw node
    ids. compute_distance_nodes below is what defines that indexing;
    any quad or dist_matrix passed around the codebase must already be
    in that index space.
  - delta(h), computed by gromov_energy, is the 4-point Gromov delta
    of a quad (Sec 1.1 of the notes): the smallest gap you get when you
    pair up the four points three different ways and compare the two
    largest pairings.
  - d_v(h) is the mean graph distance from the four nodes in h to a
    reference node v (Sec 2.1, eq. 5) -- computed inline wherever it's
    needed rather than as a standalone function here.
"""

import itertools

import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import factorized
from scipy.special import softmax


# ======================================================================
# 1. Exact Gromov hyperbolicity (four-point condition)
# ======================================================================

def gromov_energy(quads, dist_matrix):
    """
    4-point Gromov delta for a batch of quads.

    For a quad h = (x, y, z, w), pair the four points up in the three
    possible ways and sum distances within each pair:
        S1 = d(x,y) + d(z,w)
        S2 = d(x,z) + d(y,w)
        S3 = d(x,w) + d(y,z)
    Sorting S1, S2, S3 in decreasing order gives the two largest sums;
    delta(h) is half their gap. This is exactly zero for points that
    lie on a tree (no ambiguity in "which pairing is shortest") and
    grows with how far the metric deviates from tree-like.

    quads: (N, 4) array-like of node indices (see module docstring).
    dist_matrix: (n_nodes, n_nodes) all-pairs shortest-path distances,
        indexed the same way as the quads.
    Returns: (N,) array of delta(h) values, one per quad.
    """
    q = np.array(quads)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    s0 = dist_matrix[x, y] + dist_matrix[z, w]
    s1 = dist_matrix[x, z] + dist_matrix[y, w]
    s2 = dist_matrix[x, w] + dist_matrix[y, z]

    # sort the 3 pairings per quad so we can grab the top two sums
    sorted_sums = np.sort(np.stack([s0, s1, s2], axis=1), axis=1)
    largest, second_largest = sorted_sums[:, 2], sorted_sums[:, 1]
    return (largest - second_largest) / 2.0


def compute_gromov_hyperbolicity(G):
    """
    Exact (global) Gromov hyperbolicity of a graph: the max delta(h)
    over *every* 4-tuple of nodes.

    This enumerates comb(n, 4) quads, so it is only tractable up to
    roughly 300 nodes -- it exists as the ground-truth quantity that
    the local, k-hop-restricted scores in local.py are approximating.
    """
    dist_matrix, index = compute_distance_nodes(G)
    positions = list(index.values())

    quads = list(itertools.combinations(positions, 4))
    deltas = gromov_energy(quads, dist_matrix)

    return np.max(deltas)


def tuple_distance(quads, dist_matrix, target):
    """
    d_v(h) = (1/4) * sum_{j in h} d(v, j), the mean distance from the
    four nodes of each quad to the reference node `target` (eq. 5).

    quads: (N, 4) array-like of node indices, target: a node index.
    Returns: (N,) array, one mean distance per quad.
    """
    q = np.array(quads)
    return dist_matrix[q, target].mean(axis=1)


def gamma_distribution(quads, dist_matrix, target, geometric_temperature):
    """
    Locality reference distribution gamma_v (eq. 6):
        gamma_v(h_i) = softmax(-d_v(h_i) / T_geom),
    i.e. a distribution over the quads that puts more mass on those
    close to `target`. Small T_geom -> concentrated on the nearest
    quads, large T_geom -> uniform.

    Returns: (N,) array summing to 1.
    """
    w = tuple_distance(quads, dist_matrix, target)
    return softmax(-w / geometric_temperature)


# ======================================================================
# 2. All-pairs graph distances
# ======================================================================

def compute_distance_nodes(G):
    """
    All-pairs shortest-path distance matrix for G, plus the {node: index}
    mapping used to read/write into it.

    Every quad, target index, etc. passed around this codebase is in
    this index space (position in list(G.nodes())), not raw node ids --
    that's why this function's second return value matters as much as
    the matrix itself.
    """
    nodes = list(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    dist = np.full((n, n), np.inf)

    for u, lengths in nx.all_pairs_shortest_path_length(G):
        i = index[u]
        for v, d in lengths.items():
            dist[i, index[v]] = d

    return dist, index


