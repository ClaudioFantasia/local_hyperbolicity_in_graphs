import numpy as np
from scipy.special import softmax

def solve_entropic_regularization(cost_vector, T=0.01):
    """
    maximization of the objective function 
    mu @ c - mu log(mu)
    s.t \mu \in simplex of size n-1
    """
    return softmax(cost_vector / T)

def solve_l2_regularization(cost_vector, compared_distribution, lambda_reg):
    sol = compared_distribution + (cost_vector / (2 * lambda_reg))
    sol = project_to_simplex(sol)
    return sol 

def solve_KL_regularization(cost_vector, compared_distribution, T):
    compared_distribution = np.clip(compared_distribution, 1e-16 , None)
    return softmax(np.log(compared_distribution) + cost_vector / T)

def project_to_simplex(y):
    """
    Euclidean projection into the simplex
    """
    y = np.asarray(y)
    n = y.shape[0]
    u = np.sort(y)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    tau = (cssv[rho] - 1) / (rho + 1)
    x = np.maximum(y - tau, 0)
    return x

