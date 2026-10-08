"""
Closed-form solvers for the regularised simplex problem

    max_mu  <mu, cost>  -  Reg(mu)      s.t. mu in the probability simplex

which is the general shape of every optimisation in Sec 2 of the
research notes: given a per-quad cost vector (typically Gromov
energies, optionally combined with a distance-to-node penalty), find
the "importance" distribution mu over quads that trades off maximising
that cost against staying close to a reference/uniform distribution.

Each Reg(mu) below has a known closed-form maximiser, which is why
none of these need iterative optimisation.
"""

import numpy as np
from scipy.special import logsumexp, softmax


# ======================================================================
# 1. Entropic regularisation 
#
#    max_mu  <mu, cost> - T * sum_i mu_i log(mu_i)   =>   mu = softmax(cost / T)
# ======================================================================

def solve_entropic_regularization(cost_vector, T=0.01):
    """
    Closed-form mu for entropy-regularised maximisation over the simplex.
    T controls how spread out mu is: T -> 0 concentrates mu on the
    largest-cost entry (argmax); T -> inf flattens mu to uniform.
    """
    return softmax(cost_vector / T)


def entropic_objective(mu, cost_vector, T=0.01):
    """
    The objective solve_entropic_regularization maximises, evaluated at
    an arbitrary mu on the simplex:

        F(mu) = <mu, cost> - T * sum_i mu_i log(mu_i)

    i.e. the mu-weighted cost plus T times the entropy of mu. Useful to
    check how much a given mu is worth, and how far it is from the
    optimum: at mu = softmax(cost/T) the value is exactly
    T * logsumexp(cost / T), which is what entropic_optimum returns.
    """
    mu = np.clip(mu, 1e-16, None)
    return mu @ cost_vector - T * np.sum(mu * np.log(mu))


def entropic_optimum(cost_vector, T=0.01):
    """max_mu F(mu) = T * logsumexp(cost / T), the value attained by
    solve_entropic_regularization's mu."""
    return T * logsumexp(cost_vector / T)


# ======================================================================
# 2. KL regularisation against a reference distribution 
#
#    max_mu  <mu, cost> - T * KL(mu || ref)   =>   mu = softmax(log(ref) + cost / T)
#
# This is the solver behind the KL-divergence local hyperbolicity score
# in local.py: `ref` is the distance-based prior rho_v(h) that favours
# quads close to a query node, and `cost` is the Gromov energy delta(h).
# ======================================================================

def solve_KL_regularization(cost_vector, compared_distribution, T):
    """
    Closed-form mu for KL-regularised maximisation, pulling mu towards
    compared_distribution while still rewarding high-cost entries.
    compared_distribution is clipped away from 0 since log(0) = -inf.
    """
    compared_distribution = np.clip(compared_distribution, 1e-16, None)
    return softmax(np.log(compared_distribution) + cost_vector / T)


# ======================================================================
# 3. L2 regularisation
#
#    max_mu  <mu, cost> - lambda/2 * ||mu - ref||^2
#      =>   mu = project_to_simplex(ref + cost / (2 * lambda))
#
# ======================================================================

def solve_l2_regularization(cost_vector, compared_distribution, lambda_reg):
    """Closed-form (projected) mu for L2-regularised maximisation."""
    unprojected = compared_distribution + cost_vector / (2 * lambda_reg)
    return project_to_simplex(unprojected)


def project_to_simplex(y):
    """
    Euclidean projection of a vector y onto the probability simplex
    {x : x >= 0, sum(x) = 1}. Standard sort-and-threshold algorithm
    (see e.g. Duchi et al., 2008).
    """
    y = np.asarray(y)
    n = y.shape[0]
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    tau = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(y - tau, 0)


# ======================================================================
# 4. Entropic regularisation with a locality penalty
#
#    cost(h) = delta(h) - lambda_loc * d_v(h), i.e. the Gromov energy of
#    the quad discounted by its mean distance to a reference node v,
#    with lambda_loc setting how strongly distance is penalised:
#
#    max_mu  <mu, delta - lambda_loc*d_v> - T * sum_i mu_i log(mu_i)
#      =>   mu = softmax((delta - lambda_loc*d_v) / T)
#
# Same closed form as section 1, just with the shifted cost: mu now
# favours quads that are both non-tree-like *and* close to v, which is
# what makes the resulting value a *local* hyperbolicity score for v.
#
# It is also the KL case of section 2 with T_geom = T / lambda_loc:
# there mu = softmax(log gamma_v + delta/T) = softmax(delta/T - d_v/T_geom),
# which for T_geom = T/lambda_loc is exactly the mu below. The *value*
# of the two problems differs by an additive term though -- see
# local_entropic_optimum.
# ======================================================================

def solve_local_entropic_regularization(deltas, tuple_distances, T=0.01,
                                        lambda_loc=1.0):
    """
    Closed-form mu for the entropy-regularised problem with cost
    delta(h) - lambda_loc * d_v(h).

    deltas: (N,) Gromov energies of the quads (gromov_energy).
    tuple_distances: (N,) mean distance d_v(h) from each quad to the
        reference node v (tuple_distance in objectives.py).
    T: as in solve_entropic_regularization -- T -> 0 concentrates mu on
        the quad with the best delta - lambda_loc*d_v trade-off, T -> inf
        flattens.
    lambda_loc: weight of the distance penalty. 0 ignores distance
        entirely (mu = softmax(delta/T), the global section-1 problem),
        large values pin mu on the quads nearest to v whatever their
        delta.
    """
    return solve_entropic_regularization(deltas - lambda_loc * tuple_distances, T)


def local_entropic_optimum(deltas, tuple_distances, T=0.01, lambda_loc=1.0):
    """
    Optimal value of that problem,
        V* = T * logsumexp((delta - lambda_loc*d_v) / T).

    Careful when comparing V* across reference nodes: it is *not* the
    KL score plus a constant. Factoring the distance term out,

        V* = T*logsumexp(delta/T - lambda_loc*d_v/T)
           = V*_KL(T_geom = T/lambda_loc)  +  T*log Z_v,
        Z_v = sum_i exp(-lambda_loc * d_v(h_i) / T),

    and the offset T*log Z_v depends on v: it is a pure distance/volume
    term (no delta in it) that grows with how many quads sit close to v.
    lambda_loc reweights that offset but does not remove it -- at
    lambda_loc = 0 it is T*log|H| (neighborhood size), and for large
    lambda_loc it tends to -lambda_loc*min_i d_v(h_i), i.e. the score
    ends up measuring how close v's nearest quad is rather than how
    hyperbolic its neighborhood is. Subtracting T*log Z_v is exactly
    KL_score with geometric_temperature = T/lambda_loc.

    To evaluate F at some other mu, use entropic_objective with the same
    shifted cost:
    entropic_objective(mu, deltas - lambda_loc*tuple_distances, T).
    """
    return entropic_optimum(deltas - lambda_loc * tuple_distances, T)
