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
from scipy.special import softmax


# ======================================================================
# 1. Entropic regularisation (Sec 2.1, eq. 3)
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
