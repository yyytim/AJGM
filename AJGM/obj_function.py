import numpy as np
from scipy.linalg import det
def fmvnorm(p, covmat_inv, Y, mu):
    try:
        ldcovmat = np.log(det(covmat_inv))
    except:
        ldcovmat = 1e-8
    if np.linalg.det(covmat_inv) <= 0:
        ldcovmat = 1e-8
    if np.isnan(ldcovmat) or np.isinf(ldcovmat):
        ldcovmat = 1e-8
    if isinstance(mu, np.matrix):
        mu = np.array(mu)
        mu = mu.reshape(p,)
    mvnormden = -p * np.log(2 * np.pi) + ldcovmat - np.dot((Y - mu).T, np.dot(covmat_inv, (Y - mu)))
    return 0.5 * mvnormden
#Compute the likelihood function.
def objL0(Y, mu, mu_Y, S_Y, p, L, covmat_inverse, S_bar, graph, lambda1, lambda2,lambda3 ,tau, pie):
    mu_Y = np.matrix(mu_Y).reshape(-1, 1)
    log_liklyhood = 0.0
    penalty = 0.0
    n = Y.shape[0]
    for j in range(n):
        liklyhood = 0.0
        for l in range(L):
            tmp_covmat = covmat_inverse[:, (l * p):((l + 1) * p)]
            liklyhood += pie[l] * np.exp(fmvnorm(p, tmp_covmat, Y[j, :], mu[l, :]))
        log_liklyhood += np.log(liklyhood)
    for jj in range(n):
        liklyhood = 0.0
        tmp_covmat = covmat_inverse[:, (L * p):((L + 1) * p)]
        liklyhood += np.exp(fmvnorm(p, tmp_covmat, Y[jj, :], mu_Y))
        log_liklyhood += np.log(liklyhood)
    for l in range(L + 1):
        tmp_covmat = covmat_inverse[:, (l * p):((l + 1) * p)]
        tmp_covmat[np.abs(tmp_covmat) > tau] = tau
        penalty += lambda1 * (np.sum(np.abs(tmp_covmat)) - np.sum(np.abs(np.diag(tmp_covmat))))
    graph = graph  
    Num_of_edges = graph.shape[1]
    for e in range(Num_of_edges):
        l1 = graph[0, e]
        l2 = graph[1, e]
        tmp_covmat = covmat_inverse[:, int(l1 * p):int((l1 + 1) * p)] - covmat_inverse[:, int(l2 * p):int((l2 + 1) * p)]
        tmp_covmat[np.abs(tmp_covmat) > tau] = tau
        if l2==L:
            penalty += lambda3 * (np.sum(np.abs(tmp_covmat)) - np.sum(np.abs(np.diag(tmp_covmat))))
        else:
            penalty += lambda2 * (np.sum(np.abs(tmp_covmat)) - np.sum(np.abs(np.diag(tmp_covmat))))
    return [log_liklyhood - penalty , penalty]

