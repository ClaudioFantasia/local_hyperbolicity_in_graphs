import numpy as np
from scipy.special import softmax

def solve_entropic_regularization(cost_vector, T=0.01):
    """
    maximization of the objective function 
    mu @ c - mu log(mu)
    """
    return softmax(cost_vector / T)

def solve_l2_regularization(cost_vector, compared_distribution, lambda_reg):
    sol = compared_distribution + (cost_vector / (2 * lambda_reg))
    sol = project_to_simplex(sol)
    return sol 

def solve_pgd_l2_regularization(x0, u, gromov_energy, lambda_reg = 0.5, max_steps = 10):
    xk = np.copy(x0)
    lipschitz = 2 # it can be easily proved by taking the Hessian = 0
    step_rate = 1 / lipschitz
    for _ in range(max_steps):
        grad = gromov_energy - 2 * lambda_reg (xk - u)
        xk = project_to_simplex(xk + step_rate * grad)
    return xk

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

def projected_gradient_descent(cost_grad_func, x0, max_steps=10, step_size=0.25):
    """
    Generic PGD for simplex constraints.
    Note that this is for maximizing the obj function, so we have = stept_rate * grad
    cost_grad_func compute the gradient of the cost function
    """
    xk = np.copy(x0)
    for _ in range(max_steps):
        grad = cost_grad_func(xk)
        xk = project_to_simplex(xk + step_size * grad)
    return xk

