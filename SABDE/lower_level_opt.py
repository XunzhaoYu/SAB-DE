# --- basic python libs ---
from copy import deepcopy
# --- libs from Xunzhao ---
from optimization.EI import *


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Version: 2022-April-22.
Lower level optimization for handling constraints.
"""


def lower_level_init(size_upper_pop, size_lower_pop, upper_vars, lower_vars, upperbound, lowerbound,
                     elite_pop, archive, upper_F, upper_mutation_opt, lower_F, lower_mutation_opt):
    size_population = size_upper_pop * size_lower_pop
    n_upper_vars = len(upper_vars)
    n_lower_vars = len(lower_vars)
    upper_upperbound, upper_lowerbound = upperbound[upper_vars], lowerbound[upper_vars]
    lower_upperbound, lower_lowerbound = upperbound[lower_vars], lowerbound[lower_vars]

    elite_size, n_vars = np.shape(elite_pop)
    archive_size = len(archive)
    upper_pop = np.zeros((size_upper_pop, n_upper_vars))
    lower_pop = np.zeros((size_population, n_vars))
    elite_pop_up = elite_pop[:, upper_vars]
    elite_pop_lo = elite_pop[:, lower_vars]
    archive_up = archive[:, upper_vars]
    archive_lo = archive[:, lower_vars]

    # initialize upper_pop
    origin_indexes = np.random.randint(elite_size, size=size_upper_pop)
    for i in range(size_upper_pop):
        random_indexes = np.random.choice(archive_size, 2, replace=False)
        upper_pop[i] = elite_pop_up[origin_indexes[i]] + upper_F * (archive_up[random_indexes[0]] - archive_up[random_indexes[1]])
    upper_pop = np.minimum(np.maximum(upper_pop, upper_lowerbound), upper_upperbound)
    upper_pop = np.array(upper_mutation_opt.execute(upper_pop, upper_upperbound, upper_lowerbound, unique=True))

    # initialize lower_pop
    for i in range(size_upper_pop):
        shift = i * size_lower_pop
        shift_end = shift + size_lower_pop
        lower_pop[shift:shift_end, upper_vars] = upper_pop[i].copy()

        # initialize lower-level variables for all individuals in this lower-level population.
        subpop_lo = np.zeros((size_lower_pop, n_lower_vars))
        origin_indexes = np.random.randint(elite_size, size=size_lower_pop)
        for j in range(size_lower_pop):
            random_indexes = np.random.choice(archive_size, 2, replace=False)
            subpop_lo[j] = elite_pop_lo[origin_indexes[j]] + lower_F * (archive_lo[random_indexes[0]] - archive_lo[random_indexes[1]])
        subpop_lo = np.minimum(np.maximum(subpop_lo, lower_lowerbound), lower_upperbound)
        subpop_lo = np.array(lower_mutation_opt.execute(subpop_lo, lower_upperbound, lower_lowerbound, unique=True))

        lower_pop[shift:shift_end, lower_vars] = deepcopy(subpop_lo)
    return upper_pop, lower_pop


def lower_level_opt(size_upper_pop, size_lower_pop, lower_vars, upperbound, lowerbound,
                    lower_pop, lower_iteration_max, surrogates, ref_coeff,
                    lower_F, lower_mutation_opt, mode=1):
    lower_upperbound, lower_lowerbound = upperbound[lower_vars], lowerbound[lower_vars]
    size_population, n_vars = np.shape(lower_pop)

    # Recorders
    lower_pop_outputs = np.zeros((size_population, n_vars))
    feasible_list = []
    init_n_feasible = 0

    for i in range(size_upper_pop):
        shift = i * size_lower_pop
        shift_end = shift + size_lower_pop
        subpop = deepcopy(lower_pop[shift: shift_end])

        g_hat = predict_cons(subpop, surrogates, mode=mode)  # cons
        infeasible_indexes = (g_hat > ref_coeff)
        n_infeasible = len(g_hat[infeasible_indexes])
        init_n_feasible = init_n_feasible + size_lower_pop - n_infeasible  # for test #
        if n_infeasible > 0:
            full_feasible = False

            baseline = np.linspace(0, size_lower_pop, num=size_lower_pop, endpoint=False, dtype=int)
            infeasible_indexes = baseline[infeasible_indexes]
            subpop_lo = deepcopy(subpop[:, lower_vars])  # the next generation of subpop
            subpop_inf = deepcopy(subpop[infeasible_indexes, :])  # for constraint evaluation
            # --- step3: lower-level optimization. ---
            lower_iteration = 0
            while lower_iteration < lower_iteration_max:
                temp_pop = np.zeros((len(infeasible_indexes), len(lower_vars)))
                for j in range(len(infeasible_indexes)):
                    mating_indexes = np.random.choice(size_lower_pop, 3, replace=False)
                    temp_pop[j] = subpop_lo[mating_indexes[0]] + lower_F * (subpop_lo[mating_indexes[1]] - subpop_lo[mating_indexes[2]])
                subpop_lo[infeasible_indexes] = deepcopy(np.minimum(np.maximum(temp_pop, lower_lowerbound), lower_upperbound))
                subpop_lo[infeasible_indexes] = np.array(lower_mutation_opt.execute(subpop_lo[infeasible_indexes], lower_upperbound, lower_lowerbound, unique=True))

                subpop_inf[:, lower_vars] = deepcopy(subpop_lo[infeasible_indexes])

                temp_g_hat = predict_cons(subpop_inf, surrogates, mode=mode)
                new_infeasible_indexes = (temp_g_hat > ref_coeff)  # indexes in [infeasible_indexes], not subpop
                new_feasible_indexes = ~new_infeasible_indexes  # indexes in [infeasible_indexes], not subpop
                # update subpop:
                new_feasible_indexes_in_subpop = infeasible_indexes[new_feasible_indexes]   # indexes in subpop
                subpop[new_feasible_indexes_in_subpop] = deepcopy(subpop_inf[new_feasible_indexes])
                # update infeasible staff:
                infeasible_indexes = infeasible_indexes[new_infeasible_indexes]  # indexes in subpop
                subpop_inf = subpop_inf[new_infeasible_indexes]
                if len(infeasible_indexes) == 0:
                    full_feasible = True
                    break
                else:
                    lower_iteration += 1
        else:
            full_feasible = True

        # output the results of the current subpop optimization
        lower_pop_outputs[shift: shift_end] = deepcopy(subpop)
        if full_feasible is True:
            feasible_list.extend(list(range(shift, shift_end)))
        else:
            for j in range(size_lower_pop):
                if j not in infeasible_indexes:
                    feasible_list.append(j+shift)
    return lower_pop_outputs, feasible_list


def predict_cons(x, surrogate, mode=1, minimum=None, cal_EI=True):  # minimize negative EI equivalent to maximize EI.
    if mode == 1:  # ordinal surrogate
        result = surrogate.predict(x, return_mse=False)  # shape(n_samples, 1)
        return result.reshape(-1)
    else:
        results = np.zeros((len(x), 4))
        for i in range(4):
            results[:, i] = (np.abs(surrogate[i].predict(x, return_mse=False)) - 0.5).reshape(-1)
            feasible = (results[:, i] <= 0.)
            results[feasible, i] = 0
        return np.sum(results, axis=1).reshape(-1)


