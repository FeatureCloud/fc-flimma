"""
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



class ComputeNormFactors(AppState):


    def run(self) -> str or None:
        filtered_genes = self.await_data()
        self.store('counts_df', self.load('counts_df').loc[filtered_genes, :])
        self.store('lib_sizes', self.load('counts_df').sum().values)
        self.store('upper_quartile', self.load('counts_df').quantile(0.75).values)
        data_to_send = [self.load('upper_quartile'), self.load('lib_sizes')]
        self.send_data_to_coordinator(data=data_to_send, use_smpc=False)


class AggregateLibSizes(AppState):

    def run(self) -> str or None:
        clients_upper_quartiles, clients_lib_sizes = [], []
        for clients_data in self.gather_data():
            clients_upper_quartiles.append(clients_data[0])
            clients_lib_sizes.append(clients_data[1])
        lib_sizes = np.concatenate(clients_lib_sizes, axis=None)
        upper_quartiles = np.concatenate(clients_upper_quartiles, axis=None)

        quart_to_lib_size = upper_quartiles / lib_sizes
        self.broadcast_data(np.exp(np.mean(np.log(quart_to_lib_size))))


