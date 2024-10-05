from .sample_x import *
from .solveZ import *
from .obj_function import *
from .impute_function import *
from .admm_function import *
from sklearn.cluster import KMeans
import pandas as pd

def AJGM(X, Y,L ,lambda1, lambda2, lambda3,miu=0.1,tre=0.1, impute_x=False,impute_y='none', EMthreshold=0.1, MAX_iter=100):
    n = X.shape[0]
    p = X.shape[1]
    origin_mat = copy.deepcopy(X)
    allowed_methods = ['x', 'sample', 'none']
    if impute_y not in allowed_methods:
        raise ValueError(f"Invalid method for impute y. Allowed methods are: {allowed_methods}")
    if lambda3<=lambda2:
        raise ValueError('Value of lambda3 should be greater than lambda2')
    if lambda1<0 or lambda2<0 or lambda3<0 or tre<0 or miu<0:
        raise ValueError('Tuning parameters should be should be non-negative')

    #Initialize parameters
    kmeans = KMeans(n_clusters=L, n_init=1).fit(X)
    class_memship = kmeans.labels_
    memship_ini = kmeans.labels_
    center_ini = kmeans.cluster_centers_
    pie = np.bincount(class_memship) / n
    cluster_size = np.bincount(kmeans.labels_)
    ncluster = len(cluster_size)
    graph_x = np.zeros((2, ncluster))
    idx = 0
    for ni in range(ncluster):
        graph_x[:, idx] = [ni, ni+1]
        idx += 1
    mu = center_ini
    L = ncluster
    S_bar = np.zeros((p, L * p))
    for l in range(L):
        temp_X = X[memship_ini == l, :]
        mu_temp = np.mean(temp_X, axis=0) if temp_X.ndim > 1 else np.mean(temp_X)
        stemp = temp_X - mu_temp
        S_bar[:, l * p:(l + 1) * p] = (stemp.T @ stemp) / cluster_size[l]
    S_Y = np.cov(Y, rowvar=False)
    mu_Y = np.mean(Y, axis=0)
    S1 = np.zeros((p, (L + 1) * p))
    S1[:, :L * p] = S_bar
    S1[:, L * p:] = S_Y
    inputs = [S1[:, i * p:(i + 1) * p] for i in range(L + 1)]
    sol_path = admm_function(S=inputs, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,miu=miu,tre=tre
                             )
    covinv_est = np.hstack([sol_path[i] for i in range(len(sol_path))])
    covinv_output = copy.deepcopy(covinv_est)
    z = 1
    diff_1 = 100
    diff_2=100
    tau=0.1
    nplog,npenalty = objL0(X, mu, mu_Y, S_Y, p, L, covinv_est, S_bar, graph_x, lambda1, lambda2,lambda3, tau, pie)
    rho = np.zeros((L, n))
    while z < MAX_iter and abs(diff_1) > EMthreshold:
        covinv_est=covinv_output
        #Start impute function
        if impute_x:
            if z == 1:
                if impute_y=='none':
                    X = impute_function(origin_mat=origin_mat, now_mat=X, mu=mu, S_bar=S_bar,
                                        class_memship=class_memship)
                if impute_y=='x':
                    X = impute_function(origin_mat=origin_mat, now_mat=X, mu=mu, S_bar=S_bar, class_memship=class_memship)
                    Y=X
                if impute_y=='sample':
                    X = impute_function(origin_mat=origin_mat, now_mat=X, mu=mu, S_bar=S_bar, class_memship=class_memship)
                    Y=sample_multivariate_normal(X,n=X.shape[0])
            else:
                if impute_y == 'none':
                    X = impute_function(origin_mat=origin_mat, now_mat=X, mu=mu, S_bar=S_bar,
                                        class_memship=np.argmax(rho, axis=0))
                if impute_y == 'x':
                    X = impute_function(origin_mat=origin_mat, now_mat=X, mu=mu, S_bar=S_bar,
                                        class_memship=np.argmax(rho, axis=0))
                    Y = X
                if impute_y == 'sample':
                    X = impute_function(origin_mat=origin_mat, now_mat=X, mu=mu, S_bar=S_bar,
                                        class_memship=np.argmax(rho, axis=0))
                    Y = sample_multivariate_normal(X,n=X.shape[0])

        #E step
        for j in range(n):
            sm = np.sum(
                [pie[l] * np.exp(fmvnorm(p, covinv_est[:, l * p:(l + 1) * p], X[j, :], mu[l, :])) for l in range(L)])
            for l in range(L):
                rho[l, j] = pie[l] * np.exp(fmvnorm(p, covinv_est[:, l * p:(l + 1) * p], X[j, :], mu[l, :])) / sm
        #M step
        pie = np.mean(rho, axis=1)
        mu = (rho @ X) / (pie[:, None] * n + 1e-10)
        S_bar = np.zeros((p, L * p))
        for l in range(L):
            S = np.zeros((p, p))
            for j in range(n):
                S += rho[l, j] * np.outer(X[j, :] - mu[l, :], X[j, :] - mu[l, :])
            S_bar[:, l * p:(l + 1) * p] = S / (pie[l] * n + 1e-10)
        S_Y = np.cov(Y, rowvar=False)
        S2 = np.zeros((p, (L + 1) * p))
        S2[:, :L * p] = S_bar
        S2[:, L * p:] = S_Y
        inputs = [S2[:, i * p:(i + 1) * p] for i in range(L + 1)]
        #start admm
        sol_path =admm_function(S=inputs, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3,miu=miu,tre=tre)

        covinv_est = np.hstack([sol_path[i] for i in range(len(sol_path))])
        covinv_output=copy.deepcopy(covinv_est)
        plog,penalty = objL0(X, mu, mu_Y, S_Y, p, L, covinv_est, S_bar, graph_x, lambda1, lambda2,lambda3, tau, pie)
        diff_1 = abs(plog - nplog)
        diff_2=abs(penalty-npenalty)
        print(f"Iteration {z+1}: obj difference = {diff_1}, penalty difference = {diff_2}" )
        nplog = plog
        npenalty=penalty
        z +=1
        class_memship = np.argmax(rho, axis=0)
    output = {
        'pie': pie,
        'mu': mu,
        'covinv': covinv_output,
        'membership': class_memship,
        'plog':plog
    }

    return output