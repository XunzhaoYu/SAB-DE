import numpy as np
import xlwt


""" Written by Xun-Zhao Yu (yuxunzhao@gmail.com). Version: 2022-Mar-15.
Record optimization results in .xlsx file.
Designed for constrained optimization problems.
"""


class Recorder:
    def __init__(self, sheet_name='sheet1'):
        self.record_file = xlwt.Workbook()
        self.record_sheet = self.record_file.add_sheet(sheetname=sheet_name)
        # declare n_vars, n_objs, and n_cons
        self.n_vars = None
        self.n_objs = None
        self.n_cons = None
        self.style = xlwt.XFStyle()
        self.style.num_format_str = '0.0000'

    # record initial archive (X), fitness (Y), constraint (C), and performance.
    def record_init(self, X, Y, C, performance_list, performance_name_list):
        size_archive, self.n_vars = np.shape(X)
        self.n_objs = np.size(Y, 1)
        self.n_cons = np.size(C, 1)

        # initialize titles.
        self.record_sheet.write(0, 1, 'Variables')
        self.record_sheet.write(0, self.n_vars + 1, 'Objectives')
        self.record_sheet.write(0, self.n_vars + self.n_objs + 1, 'Constraints')
        for performance_index in range(len(performance_list)):
            self.record_sheet.write(0, self.n_vars + self.n_objs + self.n_cons + performance_index + 1, performance_name_list[performance_index])

        # initialize data
        for ind_index in range(size_archive-1):
            self._write_data(ind_index + 1, X[ind_index], Y[ind_index], C[ind_index])
        self.write(size_archive, X[-1], Y[-1], C[-1], performance_list)

    #  # record one evaluated solution (x, y, c) and its performance (minimum and the number of feasible solutions).
    def write(self, row_index, x, y, c, performance_list):
        self._write_data(row_index, x, y, c)
        # write performance
        for performance_index in range(len(performance_list)):
            self.record_sheet.write(row_index, self.n_vars + self.n_objs + self.n_cons + performance_index + 1, performance_list[performance_index], self.style)

    # record one evaluated solution (x, y, c):
    def _write_data(self, row_index, x, y, c):
        # write the row index
        self.record_sheet.write(row_index, 0, row_index)
        # write variables
        for dim_index in range(self.n_vars):
            self.record_sheet.write(row_index, dim_index + 1, x[dim_index], self.style)
        # write objectives
        for dim_index in range(self.n_objs):
            self.record_sheet.write(row_index, self.n_vars + dim_index + 1, y[dim_index], self.style)
        # write constraints
        for dim_index in range(self.n_cons):
            self.record_sheet.write(row_index, self.n_vars + self.n_objs + dim_index + 1, c[dim_index], self.style)

    # save record
    def record_save(self, name):
        self.record_file.save(name)


