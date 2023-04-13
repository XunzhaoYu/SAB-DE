import numpy as np


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Version: 2022-June-20.
Assign ordinal labels for ordinal regression.
This function has been customized with the constraints in the engine calibration problem. 
To use SAB-DE, please adapt this function according to the constraints in your problems.
"""


def ordinal_label(archive_constraints, infeasible, n_levels):
    size_archive = len(archive_constraints)
    n_infeasible = np.count_nonzero(infeasible)
    n_feasible = size_archive - n_infeasible

    reference = np.array([0.5, ] * 4)
    label = np.zeros(size_archive)
    infeasible_label = np.zeros(n_infeasible)
    value = 0.

    # extension coefficients.
    ec = np.abs((archive_constraints[infeasible]) / reference[0])
    ec = np.nanmax(ec, axis=1)
    ec_rank = np.argsort(ec)

    # N_o: number of ordinal levels.
    feasible_ratio = n_feasible * 1. / size_archive
    if n_feasible == 0:
        N_o = n_levels
    else:
        N_o = np.maximum(int(np.ceil(1. / feasible_ratio)), n_levels)

    delta_ratio = 1. / (N_o - 1)
    start_index = 0
    for level in range(N_o-1):
        end_index = np.minimum(np.ceil((1 + level) * delta_ratio * n_infeasible), n_infeasible-1).astype(int)
        value += delta_ratio
        infeasible_label[ec_rank[start_index:end_index]] = value
        start_index = end_index

    label[infeasible] = infeasible_label
    criterion = delta_ratio / 2
    return label, criterion, N_o
