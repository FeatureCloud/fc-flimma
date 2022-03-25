"""
    FeatureCloud Flimma Application
    Copyright 2021 Mohammad Bakhtiari. All Rights Reserved.
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

from FeatureCloud.app.engine.app import app_state, AppState, Role, SMPCOperation
import numpy as np
from States import large_n, min_prop, tol
from CustomStates.AckState import AckState
# from Acknowledge import acknowledge, get_acknowledge
# from utils import js_serializer
# from time import sleep


@app_state('CPM_Cut_Off', Role.BOTH)
class CPMCutOff(AckState):
    def register(self):
        self.register_transition('CPM_Cut_Off_Aggregation', Role.COORDINATOR)
        self.register_transition('Apply_CPM_Cut_Off', Role.PARTICIPANT)

    def run(self) -> str or None:
        self.log("Enetr CPM ")
        shared_features, cohort_effects = self.await_data()
        self.log("CPM received")
        if len(shared_features) != len(self.load('local_features')):
            self.log(f"filter_inputs_step():\t"
                     f"{len(self.load('local_features')) - len(shared_features)}"
                     f"features absent in other datasets are excluded.")
        # keep only shared features in count matrix and update local features
        counts_df = self.load('counts_df').loc[shared_features, :]
        self.store('counts_df', counts_df)
        self.store('local_features', shared_features)
        design_df = self.load('design_df')
        # receive the list of cohort names (i.e. client ids)
        # add variables modeling cohort effects to the design matrix for all but the last cohorts
        for cohort in cohort_effects:  # add n-1 columns for n cohorts
            design_df[cohort] = 1 if self.load('cohort_name') == cohort else 0
        self.store('design_df', design_df)

        self.log(cohort_effects)
        self.log(design_df.head(2))

        # compute local parameters for CPM cutoff
        self.store('variables', self.load('group1') + self.load('group2') + self.load('confounders') + cohort_effects)
        self.store('design_df', design_df.loc[:, self.load('variables')])
        self.log("###### Send out data")
        self.send_data_to_coordinator(data=self.load('counts_df').sum(axis=0).values, use_smpc=False, get_ack=True)
        if self.is_coordinator:
            self.store('clients_lib_sizes', self.gather_data(ack=True))
        self.send_data_to_coordinator(data=self.load('design_df').sum(axis=0).values.astype('int').tolist(),
                                      use_smpc=self.load('smpc_used'),
                                      get_ack=True)
        if self.is_coordinator:
            self.store('total_num_samples',
                       self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'), ack=True)
                       )
        self.send_data_to_coordinator(data=self.load('counts_df').sum(axis=1).values.astype('int').tolist(),
                                      use_smpc=self.load('smpc_used')
                                      )
        # if self.coordinator:
        #     self.store("des_counts_dfs",
        #                self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
        #                )
        # else:
        #     sleep(60)
        # self.communicate_data(data=self.load('design_df').sum(axis=0).values.astype('int'), key='sum_num_samples')
        # sleep(5)
        # self.communicate_data(data=self.load('counts_df').sum(axis=1).values.astype('int'), key='sum_count_per_feature')


        if self.is_coordinator:
            return 'CPM_Cut_Off_Aggregation'
        return 'Apply_CPM_Cut_Off'

    # def communicate_data(self, data, key):
    #     # self.log(f"{key}: {data}")
    #     # data_to_send = js_serializer.prepare(data) if self.load('smpc_used') else data
    #     self.log(f"{key}: {type(data)}")
    #
    #     data_to_send = data.tolist() if hasattr(data, "tolist") else data
    #     # if key == 'sum_num_samples' and len(self.clients) == 1 and not isinstance(data_to_send, (list, np.ndarray)):
    #     #     data_to_send = [data_to_send]
    #     #     self.log(f"{key}: {data_to_send}")
    #     self.send_data_to_coordinator(data=data_to_send, use_smpc=self.load('smpc_used'), flush=0.1)
    #     if self.is_coordinator:
    #         summed_data = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
    #         self.log(f"{key}: {summed_data}")
    #         self.store(key, summed_data)
    #
    #         for c in self.clients:
    #             acknowledge(self, c)
    #     else:
    #         get_acknowledge(self)
        # flush(self)


@app_state('CPM_Cut_Off_Aggregation', Role.COORDINATOR)
class CutOffAggregation(AppState):
    def register(self):
        self.register_transition('Apply_CPM_Cut_Off', Role.COORDINATOR)

    def run(self) -> str or None:
        self.log("HHHHHHHHHHHHHHHHHHHHHHHHHHHh")
        clients_lib_sizes = self.load('clients_lib_sizes')
        # total_num_samples, total_count_per_feature = self.load('des_counts_dfs')
        # total_num_samples, total_count_per_feature = \
        #     self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
        total_num_samples = self.load('total_num_samples')
        total_count_per_feature = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
        self.log(f"{total_num_samples}, {total_count_per_feature}")
        # total_num_samples, total_count_per_feature = \
        #     self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
        total_num_samples = np.array(total_num_samples) / len(self.clients)
        total_count_per_feature = np.array(total_count_per_feature) / len(self.clients)
        # total_num_samples = np.array(self.load('sum_num_samples')) / len(self.clients)
        # total_count_per_feature = np.array(self.load('sum_count_per_feature')) / len(self.clients)

        # clients_lib_sizes = self.gather_data()
        total_lib_sizes = np.concatenate(clients_lib_sizes)

        # define min allowed number of samples
        min_per_group_num_samples = np.min([x for x in total_num_samples if x > 0])

        if min_per_group_num_samples > large_n:
            min_per_group_num_samples = large_n + (min_per_group_num_samples - large_n) * min_prop
        self.log(f"Min. sample size: {min_per_group_num_samples}")

        genes_passed_total_count = np.where(total_count_per_feature >= (min_per_group_num_samples - tol))[0]
        self.log(f"Features in total: {len(total_count_per_feature)}")
        self.log(f"features passed total count cutoff: {len(genes_passed_total_count)}")

        median_lib_size = np.median(total_lib_sizes)
        CPM_cutoff = self.load('config')['min_count'] / median_lib_size * 1e6
        self.log(f"median lib.size: {median_lib_size}\nCPM_cutoff: {CPM_cutoff}")

        # send to clients 1) passed features
        # 2) the list of cohort variables to add (all but 1st client)
        # self.global_parameters[FlimmaGlobalParameter.CPM_CUTOFF] = CPM_cutoff
        self.broadcast_data(CPM_cutoff)
        self.store('min_per_group_num_samples', min_per_group_num_samples)
        self.store('genes_passed_total_count', genes_passed_total_count)
        return 'Apply_CPM_Cut_Off'
