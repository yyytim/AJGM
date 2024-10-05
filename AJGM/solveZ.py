import copy
import prox_tv as ptv
import numpy as np
import pywt
#To determine whether to impose a larger penalty term lambda3
def check_difference(arr,tre=0.1):
    for i in range(len(arr) - 1):
        if abs(arr[i] - arr[i + 1]) > tre:
            return False
    return True

#Update Z in ADMM
def solve_Z(Z, lambda1,lambda2,lambda3,miu=0.1,tre=0.1):
    p = len(Z[0])
    K = len(Z)
    newZ=copy.deepcopy(Z)
    for i in range(p - 1):
        for j in range(i + 1, p):
            deno1 = []
            for z in range(K):
                Z[z] = np.atleast_2d(Z[z])
                deno1.append(Z[z][i][j])
            checkdeno=deno1[:len(deno1)-1]
            weight=[]
            if check_difference(checkdeno,tre):
                for n in range(len(deno1)-2):
                    weight.append(lambda2/(abs(deno1[n+1]-deno1[n])**miu))
                weight.append(lambda3/(abs(deno1[len(deno1)-1]-deno1[len(deno1)-2])**miu))
            else:
                for n in range(len(deno1)-2):
                    weight.append(lambda2/(abs(deno1[n+1]-deno1[n])**miu))
                weight.append(lambda2/(abs(deno1[len(deno1)-1]-deno1[len(deno1)-2])**miu))
            a = ptv.tv1w_1d(deno1, weight)
            deno1 = pywt.threshold(a, lambda1, 'soft')
            for z in range(K):
                newZ[z][i][j] = deno1[z]
                newZ[z][j][i] = deno1[z]
    return newZ
