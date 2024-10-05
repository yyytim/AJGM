# Adaptive Joint Graphical Model

### Python modules required to be installed

- The package needs these Python modules to be installed beforehand.
  1. pywt, install using: "conda install pywavelets"
  2. prox_tv, install using: "pip install prox_tv"
  3. Other commonly used Python packages: numpy, pandas, scipy, scikit-learn

We recommend using Python version ≥ 3.7.
The package can only be used on Mac or Linux, not on Windows, because the "prox_tv" module is not available for Windows.

### Install and load the package
Open the file in project and install the AJGM package.
```bash
pip install .
```
Load AJGM package in Python.
```python
from AJGM import *
```

### Data description and loading

In the following example, we consider generating samples from three subtypes. The method of sample generation is consistent with the simulation study described in paper. Each subtype has the same number of genes and the same number of samples($p=200$, ${n_k} = 300$).

In AJGM, the input data should be a matrix where each row represents the gene expression values of different samples. In this example, the input matrix has 900 rows and 200 columns.

```python
import pandas as pd
data=pd.read_csv('../data/example_data.csv')
A = data.to_numpy()
```


### Parameters of AJGM

- `X`: In the AJGM code, we use X to represent the samples $x$.
- `Y`: In the AJGM code, we use Y to represent the representative samples ${x^{'}}$.
- `L`: Number of subtypes.
- `lambda1`, `lambda2`, `lambda3`: Penalty parameters $\lambda_1, \lambda_2, \lambda_3$.
- `miu`, `tre`: Parameters for adaptive weights $\gamma$, $tre$ (all default: 0.1).
- `impute_x`: Boolean, whether to perform imputation for samples X (default: False).
- `impute_y`: String, specifies imputation method for representative samples Y：
  'x': Imputation method if we set representative samples Y as X.
  'sample': Imputation method if we generate representative samples Y by sampling.
  'none': Do not perform imputation for Y (default).
- `EMthreshold`: EM algorithm convergence threshold (default: 0.1).
- `MAX_iter`: Maximum number of iterations for the EM algorithm (default: 100).

### Fitting AJGM

We first use method setting ${x_l}$ as $x_l^{'}$ to obtain representative sample Y. Since the generated data follows a mixture of multivariate normal distributions, data imputation is not necessary.

```python
res1=AJGM(X=A, Y=A, L=3, lambda1=0.01, lambda2=0.01, lambda3=0.1,miu=0.1,tre=0.1,impute_x=False,
impute_y='none', EMthreshold=0.1, MAX_iter=100)
```

We then use sampling to generation representative sample Y. The function ‘sample_multivariate_normal‘ can obtain Y through sampling. The parameter n represents the number of samples obtained through sampling.

```python
B=sample_multivariate_normal(A, n=100)
res2=AJGM(X=A, Y=B, L=3, lambda1=0.01, lambda2=0.01, lambda3=0.1,miu=0.1,tre=0.1,impute_x=False,
impute_y='none', EMthreshold=0.1, MAX_iter=100)
```

### Structure of the output

The AJGM function returns a dictionary with the following components:

- pie: Estimated parameters $\pi$.
- mu: Estimated parameters $\mu$.
- covinv: Estimated parameters $\Omega$ (The precision matrices for $K$ subtypes and an overall precision matrix). The matrix covinv is of size $p$ rows by $p*(K+1)$ columns. Columns $1$ to $p$ represent the precision matrices for subgroup 1, columns $p+1$ to $2*p$ represent subgroup 2, and so on. Columns $p*K+1$ to $p*(K+1)$ represent the precision matrix for the overall network.
- membership: Cell classification results.

### References

1. Souvik Seal, Qunhua Li, Elle Butler Basner, Laura M Saba, and Katerina Kechris. Rcfgl: Rapid condition adaptive fused graphical lasso and applica-tion to modeling brain region co-expression networks. PLoS computational biology, 19(1):e1010758, 2023.
2. Tan. Research on gene network inference method and application based on probabilistic graph modeling. PhD thesis, Central China Normal University, 2022.
