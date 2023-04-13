import numpy as np
from sklearn.decomposition import PCA
from random import random


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Version: 2022-Feb-21.
Divide decision variables into upper-level and lower-level variable via data analysis.
"""


# PCA
def data_analyze(X_feasible, print_bool=False):
    (size_elite, n_vars) = np.shape(X_feasible)
    baseline = np.linspace(0, n_vars, num=n_vars, endpoint=False, dtype=int)
    if print_bool:
        if size_elite != 0:
            print('size of archive[feasible]:', size_elite)
        else:
            print('No feasible solution')

    if size_elite < 2:
        upper_vars = baseline[np.random.choice(n_vars, n_vars//2, replace=False)]
        lower_vars = np.array([index for index in range(n_vars) if index not in upper_vars])
        if print_bool:
            print(type(upper_vars), type(lower_vars))
    else:
        pca = PCA(n_components=min(size_elite, n_vars))
        pca.fit(X_feasible)  # line 6 in Algorithm 2
        r = np.around(pca.explained_variance_ratio_, decimals=3)
        r = 1./(0.001+r)  # line 7 in Algorithm 2
        imp = np.dot(r, np.square(pca.components_))  # lines 8-10 in Algorithm 2
        if print_bool:
            print("variance", pca.explained_variance_)
            print("variance ratio", pca.explained_variance_ratio_)
            print("component", abs(pca.components_))
            print("impact", imp)
            print(' ')
        imp_ratio = imp / (imp + n_vars)  # line 11 in Algorithm 2
        print(imp_ratio)
        lower_dims = []
        for i in range(n_vars):
            if random() <= imp_ratio[i]:
                lower_dims.append(i)

        if len(lower_dims) == 0:
            lower_vars = baseline[[np.argmax(imp_ratio)]]
            print("lower=0:", lower_vars, type(lower_vars))
            upper_vars = np.array([index for index in range(n_vars) if index != lower_vars])
        elif len(lower_dims) == n_vars:
            upper_vars = baseline[[np.argmin(imp_ratio)]]
            print("upper=0:", upper_vars, type(upper_vars))
            lower_vars = np.array([index for index in range(n_vars) if index != upper_vars])
        else:
            lower_vars = baseline[lower_dims]
            upper_vars = np.array([index for index in range(n_vars) if index not in lower_vars])

    print("new upper vars:", upper_vars, " ; lower vars:", lower_vars)
    return upper_vars, lower_vars

