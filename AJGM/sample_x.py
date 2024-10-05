#Function for generating representative samples through sampling
import numpy as np
from scipy.stats import multivariate_normal
def sample_multivariate_normal(x, n=100):
    mean_vector = np.mean(x, axis=0)
    cov_matrix = np.cov(x, rowvar=False)
    new_data = multivariate_normal.rvs(mean=mean_vector, cov=cov_matrix, size=n)
    return new_data


