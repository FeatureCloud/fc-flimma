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
from FeatureCloud.engine.app import app_state, AppState, Role, SMPCOperation, LogLevel
from FeatureCloud.engine.app import State as op_state
import pandas as pd
import numpy as np
from CustomStates import ConfigState

name = 'flimma'


@app_state(name='initial', role=Role.BOTH, app_name=name)
class LocalMean(ConfigState.State):

    def register(self):
        self.register_transition('ServerInit', Role.COORDINATOR)
        self.register_transition('prepare_inputs', Role.PARTICIPANT)

    def run(self) -> str or None:
        self.lazy_init()
        self.read_config()
        self.finalize_config()
        self.store('config', self.config)
        # By default SMPC will not be used, unless end-user asks for it!
        self.store('smpc_used', self.config.get('use_smpc', False))

        flimma_counts_file_path = self.config['local_dataset']['counts']
        flimma_design_file_path = self.config['local_dataset']['design']
        normalization = self.config['normalization']
        min_count = self.config['min_count']
        min_total_count = self.config['min_total_count']
        group1 = self.config['group1']
        group2 = self.config['group2']
        confounders = self.config['confounders']
        self.store('group1', sorted([label.strip() for label in group1.split(',')]))  # assume group1 and 2 may be ,-separated lists
        self.store('group2',  sorted([label.strip() for label in group2.split(',')]))
        self.store('confounders',
                   [] if len(confounders) == 0 else
                   sorted([confounder.strip() for confounder in confounders.split(',')]))

        # Flimma specific dataset related attributes
        # self.flimma_counts_file_path = flimma_counts_file_path
        # self.flimma_design_file_path = flimma_design_file_path
        # self.local_features = []
        # self.samples = []
        # self.local_sample_count = 0
        # self.normalization = normalization
        # self.min_count = min_count
        # self.min_total_count = min_total_count
        # self.counts_df = pd.DataFrame()
        # self.design_df = pd.DataFrame()
        # self.variables = None
        # self.cohort_name = ""
        # self.norm_factors = None
        # self.lib_sizes = []
        # self.upper_quartile = []
        # self.log_cpm = pd.DataFrame()
        # self.sse = np.array([])
        # self.xt_x = np.array([])
        # self.xt_y = np.array([])
        # self.mu = np.array([])
        # self.sse = np.array([])
        # self.beta = []
        # self.cov_coefficient = None
        # self.fitted_log_counts = None
        # self.weight = None


        counts_df = pd.read_csv(flimma_counts_file_path, index_col=0, sep="\t")
        design_df = pd.read_csv(flimma_design_file_path, index_col=0, sep="\t")
        self.store('local_features', sorted(counts_df.index.values))  # features (e.g. genes)
        # ensure that the same samples are in columns of count matrix and on rows of design matrix
        self.store('samples', sorted(list(set(counts_df.columns.values).intersection(design_df.index.values))))
        self.store('local_sample_count', len(self.load('samples')))  # the number of samples found in both input files

        # ensure all target and confounder variables are in design matrix columns
        design_cols = set(design_df.columns)
        if len(set(self.load('group1')).intersection(design_cols)) == 0:
            self.log("\tClass labels %s are missing in the design matrix." % ",".join(self.load('group1')))
            self.update(state=op_state.ERROR)
        if len(set(self.load('group2')).intersection(design_cols)) == 0:
            self.log("\t Class labels %s are missing in the design matrix." % ",".join(self.load('group2')))
            self.update(state=op_state.ERROR)
        missing_conf_variables = set(self.load('confounders')).difference(set(design_cols))
        if len(missing_conf_variables) > 0:
            self.log(
                "\tConfounder variable(s) are missing in the design matrix: %s." % ",".join(missing_conf_variables))
            self.update(state=op_state.ERROR)

        # sort and save inputs

        self.store('counts_df', counts_df.loc[self.local_features, self.samples])
        self.store('variables', self.load('group1') + self.load('group2') + self.load('confounders'))  # list of all relevant variables
        self.store('design_df', design_df.loc[self.load('samples'), self.load('variables')])

        self.send_data_to_coordinator(data=self.load('local_sample_count'), use_smpc=self.load('smpc_used'))
        self.log("\tCount matrix: %s features x %s samples" % self.load('counts_df').shape)

        # name cohort
        self.store('cohort_name', "Cohort_" + self.username ) # name is unique -- user can join the project only once

        # send sample counts to the server
        # self.local_parameters[FlimmaLocalParameter.SAMPLE_COUNT] = self.local_sample_count
        self.store('norm_factors', np.ones(self.counts_df.shape[1]))

        # prepare input step

        # send the list of features and cohort names to the server
        # self.local_parameters[FlimmaLocalParameter.FEATURES] = self.local_features
        # self.local_parameters[FlimmaLocalParameter.COHORT_NAME] = self.cohort_name
        self.send_data_to_coordinator(data=self.load('local_features'), use_smpc=False)
        self.send_data_to_coordinator(data=self.load('cohort_name'), use_smpc=False)
        if self.coordinator:
            return 'ServerInit'
        return 'terminal'

@app_state('ServerInit', Role.COORDINATOR)
class ServerInit(AppState):
    def register(self):
        self.register_transition('prepare_inputs', Role.COORDINATOR)

    def run(self) -> str or None:
        if not hasattr(self, 'execution_counter'):
            self.execution_counter = 0
        if self.execution_counter == 0:
            self.global_sample_count = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=self.load('smpc_used'))
            self.execution_counter += 1
            return 'ServerInit'
        elif self.execution_counter == 1:
            self.cohort_names = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=False)
            self.store('cohort_effects', sorted(self.cohort_names)[:-1])  # will add confounders for all but one cohorts
            self.execution_counter += 1
            return 'ServerInit'
        feature_lists = self.aggregate_data(operation=SMPCOperation.ADD, use_smpc=False)

        # self.cohort_names = (self.local_parameters, FlimmaLocalParameter.COHORT_NAME)
        # self.cohort_effects = sorted(self.cohort_names)[:-1]  # will add confounders for all but one cohorts

        # collect features from clients and keep only features shared by all clients
        # feature_lists = client_parameters_to_list(self.local_parameters, FlimmaLocalParameter.FEATURES)
        # for f in feature_lists:
        #     print(len(f))
        shared_features = set(feature_lists[0])
        for feature_list in feature_lists[1:]:
            shared_features = shared_features.intersection(set(feature_list))
        self.store('gene_name_list', sorted(list(shared_features)))
        self.store('n_features', len(self.gene_name_list))

        # testprints
        print("#############")
        print("Total samples:", self.global_sample_count)
        print("Shared features:", self.load('gene_name_list')[:3], "...", len(self.load('gene_name_list')), "features")
        print("Joined cohorts:", self.cohort_names)
        print("Cohort effects added:", self.load('cohort_effects'))
        print("############")

        # send to clients 1) shared features 2) the list of cohort names
        # self.global_parameters[FlimmaGlobalParameter.FEATURES] = self.gene_name_list
        # self.global_parameters[FlimmaGlobalParameter.COHORT_EFFECTS] = self.cohort_effects
        data_to_send = [self.gene_name_list, self.cohort_effects]
        self.broadcast_data(data=data_to_send)

        return 'terminal'


# @app_state('prepare_inputs', Role.BOTH)
# class prepare_inputs(AppState):
#     def register(self):
#         self.register_transition('', Role.COORDINATOR)
#
#     def run(self) -> str or None:




