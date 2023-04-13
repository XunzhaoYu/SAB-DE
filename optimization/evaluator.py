import numpy as np
import matlab.engine


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Version: 2022-Mar-15.
The evaluator of real-world problems.
"""


class Evaluator:
    def __init__(self, upperbound, lowerbound):
        self.upperbound = upperbound
        self.lowerbound = lowerbound

    def population_evaluation_Matlab(self, population, is_normalized_data=False):
        if is_normalized_data:
            population = population * (self.upperbound - self.lowerbound) + self.lowerbound
        print('start matlab')
        eng = matlab.engine.start_matlab()
        prediction = eng.predict(matlab.double(list(population.reshape(1, -1)[0])), async=True)  # nargout=0
        res = prediction.result()
        eng.quit()
        print('quit matlab')
        cons_and_obj = np.around(np.array(res), decimals=4)
        obj = cons_and_obj[:, -1]
        cons = cons_and_obj[:, :-1]
        return obj.reshape(-1, 1), cons
