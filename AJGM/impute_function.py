import numpy as np
#Imputation method for scRNA-seq data.
def impute_function(origin_mat, now_mat, mu, S_bar, class_memship):
    n, p = origin_mat.shape
    for i in range(n):
        l = class_memship[i]
        mean_vec = mu[l, :]
        cov_mat = S_bar[:, l * p:(l + 1) * p]
        filled_value = np.random.multivariate_normal(mean_vec, cov_mat)
        for j in range(p):
            if origin_mat[i, j] == 0:
                now_mat[i, j] = filled_value[j]
    return now_mat