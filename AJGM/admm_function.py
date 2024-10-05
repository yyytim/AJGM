import numpy as np
from .solveZ import *
#Function for implementing the ADMM algorithm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
def admm_function(S, lambda1, lambda2, lambda3, rho=1, maxiter=100, tol=1e-3,miu=0.1,tre=0.1):
    p = S[0].shape[1]
    K = len(S)
    S = [S[k] for k in range(K)]
    weights = np.ones(K)
    theta = [np.diag(1 / np.diag(S[k])) for k in range(K)]
    Z = [np.zeros((p, p)) for k in range(K)]
    W = [np.zeros((p, p)) for k in range(K)]
    iter = 0
    diff_value = 10
    while iter < maxiter and diff_value > tol:
        theta_prev = theta.copy()
        for k in range(K):
            edecomp = np.linalg.eigh(S[k] - rho * Z[k] / weights[k] + rho * W[k] / weights[k])
            D = edecomp[0]
            V = edecomp[1]
            D2 = weights[k] / (2 * rho) * (-D + np.sqrt(D ** 2 + 4 * rho / weights[k]))
            theta[k] = V @ np.diag(D2) @ V.T
        A = [theta[k] + W[k] for k in range(K)]
        Z = solve_Z(A, lambda1, lambda2, lambda3,miu,tre)
        for k in range(K):
            W[k] = W[k] + (theta[k] - Z[k])
        iter += 1
        diff_value = sum(np.sum(np.abs(theta[k] - theta_prev[k])) / np.sum(np.abs(theta_prev[k])) for k in range(K))
    return theta

