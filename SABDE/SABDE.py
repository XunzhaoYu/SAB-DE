# --- basic python libs ---
from time import time
import xlrd
# --- SAB-DE libs ---
from SABDE.data_analysis import *
from SABDE.ordinal_label import *
from SABDE.lower_level_opt import *   # lower_level_init, lower_level_opt
# --- surrogate modeling ---
from models.pydacefit.dace import *
from models.pydacefit.regr import *
from models.pydacefit.corr import *
# --- optimization libraries ---
from optimization.operators.mutation_operator import *
from optimization.EI import *
from optimization.evaluator import *
# --- tools ---
from tools.recorder import *

desired_width = 160
np.set_printoptions(linewidth=desired_width)
np.set_printoptions(precision=4, suppress=True)
np.set_printoptions(threshold=np.inf)


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Version: 2022-April-22.
SAB-DE: Surrogate-Assisted Bilevel Differential Evolution
"""


class SABDE:
    def __init__(self, config, name, init_path=None):
        self.config = deepcopy(config)
        self.init_path = init_path
        # --- problem setups ---
        self.fitness_function = name
        self.n_vars = self.config['x_dim']
        self.n_objs = self.config['y_dim']
        self.n_cons = self.config['c_dim']
        self.upperbound = np.array(self.config['x_upperbound'])
        self.lowerbound = np.array(self.config['x_lowerbound'])
        
        self.upper_to_lower_bounds = self.upperbound - self.lowerbound
        self.evaluator = Evaluator(self.upperbound, self.lowerbound)

        # --- surrogate setups ---
        self.fit_criterion = self.n_vars
        self.n_models = 2
        self.n_near_levels = 1
        self.n_levels = self.config['n_levels']
        self.dace_training_iteration_init = self.config['dace_training_iteration_init']
        self.dace_training_iteration = self.config['dace_training_iteration']
        self.coe_range = self.config['coe_range']
        self.exp_range = self.config['exp_range']

        # --- optimization algorithm setups ---
        # evaluation budget
        self.evaluation_init = self.config['evaluation_init']
        self.evaluation_max = self.config['evaluation_max']
        # bi-level: population sizes and iterations:
        self.size_elite_pop = self.config['size_elite_pop']
        self.size_upper_pop = self.config['size_upper_pop']
        self.size_lower_pop = self.config['size_lower_pop']
        self.size_population = self.size_upper_pop * self.size_lower_pop

        self.lower_iteration_max = self.config['lower_iteration_max']
        # DE operator settings
        self.upper_F = self.config['upper_F']
        self.upper_mutation_opt = Polynomial(distribution_index=20)
        self.lower_F = self.config['lower_F']
        self.lower_mutation_opt = Polynomial(distribution_index=20)

        # reproduction
        self.n_reproduction = 1

        # --- variables declarations (see variable_init()) ---
        self.time = None
        self.iteration = None
        # --- --- archive and surrogate variables --- ---
        self.archive = None
        self.archive_fitness = None
        self.archive_constraints = None
        self.size_archive = 0
        self.theta = np.zeros((self.n_models, 2 * self.n_vars))
        self.surrogates = []
        # --- --- bi-level variables --- ---
        self.upper_vars = np.array([0, 2, 3, 4])
        self.n_upper_vars = len(self.upper_vars)
        self.lower_vars = np.array([index for index in range(self.n_vars) if index not in self.upper_vars])
        self.n_lower_vars = self.n_vars - self.n_upper_vars
        self.upper_pop = None  # np.zeros((self.size_upper_pop, self.n_upper_vars))  # lhs(self.n_upper_vars, self.size_upper_pop)  #10*4
        self.lower_pop = None  # np.zeros((self.size_population, self.n_vars))  # 400*6
        self.minimum = self.solution = None
        self.label = self.reference_point = None  # self.rp_index_for_pf = None
        # --- recorder ---
        self.recorder = None

    """
    Initialization methods:
    set variables and surrogate for a new iteration.
    """
    def variable_init(self, current_iteration):
        """
        initialize surrogate, reset all variables.
        """
        self.time = time()
        self.iteration = current_iteration
        # --- archive and surrogate variables ---
        self.archive, self.archive_fitness, self.archive_constraints = self._archive_init()
        self.size_archive = len(self.archive)
        self.surrogates = []
        for i in range(self.n_models):
            self.theta[i] = np.append(np.ones(self.n_vars) * np.mean(self.coe_range), np.ones(self.n_vars) * np.mean(self.exp_range))
            temp_surrogate = DACE(regr=regr_constant, corr=corr_gauss2, theta=self.theta[i],
                                  thetaL=np.append(np.ones(self.n_vars) * self.coe_range[0], np.ones(self.n_vars) * self.exp_range[0]),
                                  thetaU=np.append(np.ones(self.n_vars) * self.coe_range[1], np.ones(self.n_vars) * self.exp_range[1]))
            self.surrogates.append(temp_surrogate)
        # --- --- bi-level variables --- ---
        self.upper_vars = np.array([0, 2, 3, 4])
        self.n_upper_vars = len(self.upper_vars)
        self.lower_vars = np.array([index for index in range(self.n_vars) if index not in self.upper_vars])
        self.n_lower_vars = self.n_vars - self.n_upper_vars
        self.upper_pop = None
        self.lower_pop = None

        self.minimum = max(self.archive_fitness)[0]  # result for the single objective case.
        self.solution = None

        self.label = np.zeros((self.n_models, self.size_archive))
        self.reference_point = np.array([0.5, ] * self.n_cons)
        # --- --- data --- ---
        self.feasible = (np.abs(self.archive_constraints[:, 0]) <= 0.5) & (np.abs(self.archive_constraints[:, 1]) <= 0.5) & \
                        (np.abs(self.archive_constraints[:, 2]) <= 0.5) & (np.abs(self.archive_constraints[:, 3]) <= 0.5)
        self.infeasible = ~self.feasible
        self.feasible_archive = self.archive[self.feasible]
        self.n_feasible = len(self.feasible_archive)
        print("number of feasible: {:d}".format(self.n_feasible))
        if self.n_feasible != 0:
            self.minimum = np.min(self.archive_fitness[self.feasible])
        if self.n_feasible <= self.fit_criterion:
            self.fitness_theta = np.zeros((self.n_cons, 2 * self.n_vars))
            self.fitness_surrogates = []
            print("train {:d} fitness surrogates for special case".format(self.n_cons))
            for i in range(self.n_cons):
                self.fitness_theta[i] = np.append(np.ones(self.n_vars) * np.mean(self.coe_range), np.ones(self.n_vars) * np.mean(self.exp_range))
                temp_surrogate = DACE(regr=regr_quadratic, corr=corr_gauss2, theta=self.fitness_theta[i],
                                      thetaL=np.append(np.ones(self.n_vars) * self.coe_range[0], np.ones(self.n_vars) * self.exp_range[0]),
                                      thetaU=np.append(np.ones(self.n_vars) * self.coe_range[1], np.ones(self.n_vars) * self.exp_range[1]))
                self.fitness_surrogates.append(temp_surrogate)

        print("optimum:", self.minimum)
        # --- recorder ---
        self.recorder = Recorder('engine')
        self.recorder.record_init(self.archive, self.archive_fitness, self.archive_constraints, [self.minimum, self.n_feasible], ['BSFC', 'n_feasible'])

    # Invoked by self.variable_init()
    def _archive_init(self):
        if self.init_path is None:
            archive = lhs(self.n_vars, samples=self.evaluation_init)
            archive_fitness, archive_constraints = self.evaluator.population_evaluation_Matlab(archive, True)
        else:
            src_path = self.init_path + self.iteration + ".xlsx"
            src_file = xlrd.open_workbook(src_path)
            src_sheets = src_file.sheets()
            src_sheet = src_sheets[0]
            archive = np.zeros((self.evaluation_init, self.n_vars), dtype=float)
            obj_and_con = np.zeros((self.evaluation_init, self.n_objs + self.n_cons), dtype=float)
            for index in range(self.evaluation_init):
                new_row = src_sheet.row_values(index)
                archive[index, :] = new_row[:self.n_vars]
                obj_and_con[index, :] = new_row[self.n_vars + 1: self.n_vars + self.n_objs + self.n_cons + 1]
            obj_and_con = np.around(obj_and_con, decimals=4)
            archive_fitness = obj_and_con[:, self.n_cons]
            archive_constraints = obj_and_con[:, :self.n_cons]
        return archive, archive_fitness.reshape(-1, self.n_objs), self._constraint_preprocess(archive_constraints)

    def _constraint_preprocess(self, fitness):
        constraints = fitness - 0.5
        return constraints

    def run(self, current_iteration):
        self.variable_init(current_iteration)
        # Iterative procedure:
        self.last_n_levels = 0
        while self.size_archive < self.evaluation_max:
            print(" ")
            print(" --- --- Dividing Variables ... --- ---")
            self.upper_vars, self.lower_vars = data_analyze(self.feasible_archive, self.archive[self.infeasible])
            self.n_upper_vars, self.n_lower_vars = len(self.upper_vars), len(self.lower_vars)

            print(" --- --- Labeling and Training Kriging model... --- ---")
            self.label = np.zeros((self.n_models, self.size_archive))
            self.label[0] = self.archive_fitness[:, 0]  # upper level
            self.label[1], self.ref_coeff, n_levels = ordinal_label(self.archive_constraints, self.infeasible, self.n_levels)

            if self.size_archive == self.evaluation_init:
                for i in range(self.n_models):
                    self.surrogates[i].fit(self.archive, self.label[i], self.dace_training_iteration_init*10)
                    self.theta[i] = self.surrogates[i].model["theta"]

            self.surrogates[0].fit(self.archive, self.label[0], self.dace_training_iteration)
            self.theta[0] = self.surrogates[0].model["theta"]
            if self.last_n_levels != n_levels:
                self.surrogates[1].fit(self.archive, self.label[1], self.dace_training_iteration_init)
            else:
                self.surrogates[1].fit(self.archive, self.label[1], self.dace_training_iteration)
            self.theta[1] = self.surrogates[1].model["theta"]
            self.last_n_levels = n_levels

            if self.n_feasible <= self.fit_criterion:
                for i in range(self.n_cons):
                    self.fitness_surrogates[i].fit(self.archive, self.archive_constraints[:, i], self.dace_training_iteration)
                    self.fitness_theta[i] = self.fitness_surrogates[i].model["theta"]
            print("updated theta:")
            print(self.theta)

            # OREA with bi-level: for every lower level population, run a GA.
            infeasible_division = 3
            print(" --- --- Updating Upper Pop ... --- ---")
            if self.size_archive == self.evaluation_init:
                size_ordinal = np.floor((self.size_archive - self.n_feasible) // infeasible_division).astype(int)
                size_init = np.maximum(self.size_elite_pop, self.n_feasible + size_ordinal * self.n_near_levels)
                order_label1 = np.argsort(self.label[1])[: size_init]
                elite_indexes = order_label1[np.argsort(self.label[0, order_label1])[:self.size_elite_pop]]
                self.elite_pop = self.archive[elite_indexes, :]
            else:
                criterion = 1. / infeasible_division * self.n_near_levels
                print("last cons: {:.4f} <= {:.4f}? fitness: {:.4f}.".format(self.label[1, -1], criterion, self.label[0, -1]))
                if self.label[1, -1] <= criterion:  # continue update
                    replacement_index = np.random.choice(self.size_elite_pop, 1)[0]
                    replacement_fit = self.label[0, elite_indexes[replacement_index]]
                    if self.label[0, -1] < replacement_fit:
                        elite_indexes[replacement_index] = self.size_archive - 1
                        self.elite_pop = self.archive[elite_indexes, :]
                        print("updated elite indexes {:d}: {:f} -> {:f}".format(replacement_index, replacement_fit, self.label[0, -1]))
            print(self.label[0, elite_indexes])

            print(" --- --- Bi-Level Optimization Searching ... --- ---")
            new_point = np.zeros((self.n_reproduction, self.n_vars))
            self.upper_pop, self.lower_pop = lower_level_init(
                                self.size_upper_pop, self.size_lower_pop, self.upper_vars, self.lower_vars, self.upperbound, self.lowerbound,
                                self.elite_pop, self.archive, self.upper_F, self.upper_mutation_opt, self.lower_F, self.lower_mutation_opt)

            if self.n_feasible > self.fit_criterion:
                self.lower_pop, feasible_list = lower_level_opt(
                                    self.size_upper_pop, self.size_lower_pop, self.lower_vars, self.upperbound, self.lowerbound,
                                    self.lower_pop, self.lower_iteration_max, self.surrogates[1], self.ref_coeff,
                                    self.lower_F, self.lower_mutation_opt, mode=1)
            else:
                self.lower_pop, feasible_list = lower_level_opt(
                                    self.size_upper_pop, self.size_lower_pop, self.lower_vars, self.upperbound, self.lowerbound,
                                    self.lower_pop, self.lower_iteration_max, self.fitness_surrogates, self.ref_coeff,
                                    self.lower_F, self.lower_mutation_opt, mode=self.n_cons)
            n_feasible_pop = len(feasible_list)

            similarity_flag = True
            if n_feasible_pop > 0:
                feasible_pop = self.lower_pop[feasible_list]
                h_hat = np.zeros(n_feasible_pop)
                for i in range(n_feasible_pop):
                    h_hat[i] = self.predict_EI(feasible_pop[i], self.surrogates[0], minimum=self.minimum)
                rank1 = np.argsort(h_hat)
                for i in range(n_feasible_pop):
                    new_point_index = rank1[i]
                    similarity = np.sum(np.square(feasible_pop[new_point_index] - self.archive), axis=1)
                    if np.min(similarity) > 1e-5:
                        new_point = ((feasible_pop[new_point_index]).copy()).reshape(self.n_reproduction, -1)
                        similarity_flag = False
                        break
            if (n_feasible_pop <= 0) or similarity_flag:
                if self.n_feasible > self.fit_criterion:
                    g_hat = predict_cons(self.lower_pop, self.surrogates[1], mode=1)
                else:
                    g_hat = predict_cons(self.lower_pop, self.fitness_surrogates, mode=self.n_cons)
                rank2 = np.argsort(g_hat)
                for i in range(len(self.lower_pop)):
                    new_point_index = rank2[i]
                    similarity = np.sum(np.square(self.lower_pop[new_point_index] - self.archive), axis=1)
                    if np.min(similarity) > 1e-5:
                        new_point = ((self.lower_pop[new_point_index]).copy()).reshape(self.n_reproduction, -1)
                        similarity_flag = False
                        break
            if similarity_flag:
                print("similar to a point that already in the archive")
                if n_feasible_pop > 0:
                    new_point = ((feasible_pop[rank1[0]]).copy()).reshape(self.n_reproduction, -1)
                else:
                    new_point = ((self.lower_pop[rank2[0]]).copy()).reshape(self.n_reproduction, -1)
                    
            new_obj, new_cons = self.evaluator.population_evaluation_Matlab(new_point, True)
            print("new point & objective:")
            print(new_point[0], "  ", new_cons[0], "  ", new_obj[0])

            # --- step6: update archive and progress. ---
            self.archive = np.append(self.archive, new_point, axis=0)
            self.archive_fitness = np.append(self.archive_fitness, new_obj, axis=0)
            self.archive_constraints = np.append(self.archive_constraints, self._constraint_preprocess(new_cons), axis=0)

            self.feasible = (np.abs(self.archive_constraints[:, 0]) <= 0.5) & (np.abs(self.archive_constraints[:, 1]) <= 0.5) & \
                            (np.abs(self.archive_constraints[:, 2]) <= 0.5) & (np.abs(self.archive_constraints[:, 3] <= 0.5))
            self.infeasible = ~self.feasible
            self.feasible_archive = self.archive[self.feasible]
            self.n_feasible = len(self.feasible_archive)
            if self.n_feasible != 0:
                self.minimum = np.min(self.archive_fitness[self.feasible])
            self._progress_update(self.size_archive + 1, new_point, new_obj, new_cons, self.minimum, self.n_feasible, self.iteration, self.time)
            self.size_archive += self.n_reproduction

        # --- step7: when evaluation budget is running out, output the result.  ---
        if self.n_feasible != 0:
            solution_index = np.argmin(self.archive_fitness[self.feasible])
            self.solution, self.minimum = self.feasible_archive[solution_index], self.minimum
        else:
            self.solution, self.minimum = None, None

    def predict_EI(self, x, surrogate, minimum=None):  # minimize negative EI equivalent to maximize EI.
        x = np.array(x).reshape(1, -1)
        mu_hat, sigma2_hat = surrogate.predict(x, return_mse=True)
        if sigma2_hat <= 0.:
            ei = 0.  # minimum - mu_hat
        else:
            ei = EI(minimum=minimum, mu=mu_hat, sigma=np.sqrt(sigma2_hat))
        return -ei

    def _progress_update(self, row_index, x, y, c, minimum, n_feasible, iteration, curr_time):
        # write the record
        for i in range(len(x)):
            self.recorder.write(row_index + i, x[i], y[i], c[i], [minimum, n_feasible])
        self.recorder.record_save("SAB-DE" + iteration + ".xlsx")

        # print results
        t = time() - curr_time
        print("SAB-DE, Evaluation Count: {:d}.  Total time: {:.0f} mins, {:.2f} secs.".format(row_index + len(x) - 1, t // 60, t % 60))
        print("Feasible archive size: {:d}, optimum: {:.4f}".format(n_feasible, minimum))

    def get_result(self):
        self.recorder.record_save("SAB-DE-(1_3)-" + self.iteration + " " + str(self.minimum) + "(" + str(self.n_feasible) + ") " +
                                  str((time() - self.time) // 60) + "mins.xlsx")
        return self.solution, self.minimum
