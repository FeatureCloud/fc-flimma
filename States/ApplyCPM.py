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
from States import tol
from utils import js_serializer

@app_state('Apply_CPM_Cut_Off', Role.BOTH)
class ApplyCPM(AppState):
    def register(self):
        self.register_transition('Aggregate_Gene_Names', Role.COORDINATOR)
        self.register_transition('Compute_Norm_Factors', Role.PARTICIPANT)

    def run(self) -> str or None:
        cpm_cutoff = self.await_data()
        lib_sizes = self.load('counts_df').sum(axis=0)
        cpm = self.load('counts_df') / (lib_sizes * self.load('norm_factors') + 1) * 10 ** 6
        cpm_cutoff_sample_count = cpm[cpm >= cpm_cutoff].apply(lambda x: sum(x.notnull().values), axis=1).values
        data_to_send = js_serializer.prepare(cpm_cutoff_sample_count) if self.load('smpc_used') else cpm_cutoff_sample_count
        self.send_data_to_coordinator(data=data_to_send, use_smpc=self.load('smpc_used'))
        if self.is_coordinator:
            return 'Aggregate_Gene_Names'
        return 'Compute_Norm_Factors'


@app_state('Aggregate_Gene_Names', Role.COORDINATOR)
class AggregateGeneNames(AppState):
    def register(self):
        self.register_transition('Compute_Norm_Factors', Role.COORDINATOR)

    def run(self) -> str or None:
        summ_cpm_cutoff_sample_count = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
        total_cpm_cutoff_sample_count = np.array(summ_cpm_cutoff_sample_count) / len(self.clients)
        genes_passed_cpm_cutoff = \
            np.where(total_cpm_cutoff_sample_count > self.load('min_per_group_num_samples') - tol)[0]
        self.log(f"features passed CPM cutoff: {len(genes_passed_cpm_cutoff)}")
        intersect = sorted(list(set(genes_passed_cpm_cutoff).intersection(set(self.load('genes_passed_total_count')))))
        self.store('gene_name_list', np.array(self.load('gene_name_list'))[intersect].tolist())
        self.log(f"features passed both cutoffs: {len(self.load('gene_name_list'))}")
        self.broadcast_data(self.load('gene_name_list'))
        return 'Compute_Norm_Factors'
