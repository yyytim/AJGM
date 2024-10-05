from AJGM import *
import pandas as pd
data = pd.read_csv('../data/example_data.csv')
A = data.to_numpy()
res1=AJGM(X=A, Y=A, L=3, lambda1=0.01, lambda2=0.01, lambda3=0.1,miu=0.1,tre=0.1,impute_x=False,
impute_y='none', EMthreshold=0.1, MAX_iter=100)
B=sample_multivariate_normal(A, n=100)
res2=AJGM(X=A, Y=B, L=3, lambda1=0.01, lambda2=0.01, lambda3=0.1,miu=0.1,tre=0.1,impute_x=False,
impute_y='none', EMthreshold=0.1, MAX_iter=100)


