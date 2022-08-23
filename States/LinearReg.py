""""
    FeatureCloud Flimma Application
    Copyright 2022 Mohammad Bakhtiari. All Rights Reserved.
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

from FeatureCloud.app.engine.app import app_state, AppState, Role, SMPCOperation, LogLevel
import numpy as np
from scipy import linalg
from scipy.interpolate import interp1d
from utils import js_serializer
from States import get_k_n
from CustomStates.AckState import AckState


@app_state('Linear_Regression', Role.BOTH)
class LinearRegression(AckState):
    def __init__(self):
        self.weighted = False

    def register(self):
        self.register_transition('Aggregate_Reg_Params', Role.COORDINATOR)
        self.register_transition('SSE', Role.PARTICIPANT)

    def run(self) -> str or None:
        if self.weighted:
            print("Weighted Regression")
            py_lowess = self.await_data()
            self.weight_step(py_lowess)
        else:
            print("Regression")
            f = self.await_data()
            print("F received!!!")
            self.store('norm_factors', self.load('upper_quartile') / self.load('lib_sizes') / f)
        self.compute_log_cpm()
        self.compute_linear_regression_parameters()
        self.weighted = not self.weighted
        self.communicate_data()
        if self.is_coordinator:
            return 'Aggregate_Reg_Params'
        return 'SSE'

    def communicate_data(self):
        data_to_send = js_serializer.prepare(self.load('xt_x')) if self.load('smpc_used') else self.load('xt_x')
        self.send_data_to_coordinator(data=data_to_send, use_smpc=self.load('smpc_used'), get_ack=True)

        if self.is_coordinator:
            self.store('sum_xt_x',
                       self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'), ack=True)
                       )
        data_to_send = js_serializer.prepare(self.load('xt_y')) if self.load('smpc_used') else self.load('xt_y')
        self.send_data_to_coordinator(data=data_to_send, use_smpc=self.load('smpc_used'))

    def compute_log_cpm(self, add=0.5, log2=True):
        log_cpm = self.load('counts_df').applymap(lambda x: x + add)
        log_cpm = log_cpm / (self.load('lib_sizes') * self.load('norm_factors') + 1) * 10 ** 6
        if log2:
            log_cpm = log_cpm.applymap(lambda x: np.log2(x))
        self.store('log_cpm', log_cpm)

    def compute_linear_regression_parameters(self):

        x_matrix = self.load('design_df').values
        y_matrix = self.load('log_cpm').values  # Y - logCPM (samples x genes)
        n = y_matrix.shape[0]  # genes
        k = self.load('design_df').shape[1]  # conditions
        xt_x = np.zeros((n, k, k))
        xt_y = np.zeros((n, k))
        mu = np.zeros(y_matrix.shape)

        if self.weighted:
            w_square = np.sqrt(self.load('weight'))
            y_matrix = np.multiply(y_matrix, w_square)  # algebraic multiplications by W

        # linear models for each row
        for i in range(0, n):  #
            y = y_matrix[i, :]
            if self.weighted:
                x_w = np.multiply(x_matrix, w_square[i, :].reshape(-1, 1))  # algebraic multiplications by W
                xt_x[i, :, :] = x_w.T @ x_w
                xt_y[i, :] = x_w.T @ y
            else:
                xt_x[i, :, :] = x_matrix.T @ x_matrix
                xt_y[i, :] = x_matrix.T @ y
        self.store('xt_x', xt_x)
        self.store('xt_y', xt_y)
        self.store('mu', mu)

    def weight_step(self, py_lowess):
        '''Converts fitted logCPM back to fitted log-counts.'''
        fitted_counts = (2 ** self.load(
            'mu').T) * 10 ** -6  # fitted logCPM -> fitted CPM -> fitted counts/norm_lib_size
        norm_lib_sizes = self.load('lib_sizes') * self.load('norm_factors') + 1
        fitted_counts = np.multiply(fitted_counts, norm_lib_sizes.reshape(-1, 1)).T
        fitted_log_counts = np.log2(fitted_counts)

        lo = interp1d(py_lowess[:, 0], py_lowess[:, 1], kind="nearest", fill_value="extrapolate")
        self.store('weight', lo(fitted_log_counts) ** -4)


@app_state('Aggregate_Reg_Params', Role.COORDINATOR)
class AggregateRegression(AppState):
    def register(self):
        self.register_transition('SSE', Role.COORDINATOR)

    def run(self) -> str or None:
        global_xt_x = np.array(self.load('sum_xt_x')) / len(self.clients)
        self.log(f"XTX: {global_xt_x.shape}")
        sum_xt_y = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
        global_xt_y = np.array(sum_xt_y) / len(self.clients)
        self.log(f"XTY: {global_xt_y.shape}")
        k, n = get_k_n(self.load('server_vars'), self.load('confounders'), self.load('cohort_names'),
                       self.load('gene_name_list'))
        self.log(f"N={n}\nK={k}, ")
        beta = np.zeros((n, k))
        rank = np.ones(n) * k
        std_unscaled = np.zeros((n, k))

        for i in range(0, n):
            inv_xt_x = linalg.inv(global_xt_x[i, :, :])
            beta[i, :] = inv_xt_x @ global_xt_y[i, :]
            std_unscaled[i, :] = np.sqrt(np.diag(inv_xt_x))
        self.broadcast_data(beta)
        self.store('beta', beta)
        self.store('std_unscaled', std_unscaled)
        return 'SSE'
