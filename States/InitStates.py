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
from FeatureCloud.app.engine.app import AppState, Role
from CustomStates.AckState import AckState
from FeatureCloud.app.engine.app import State as op_state
import numpy as np
from CustomStates import ConfigState
from utils import readfiles


class LocalMean(ConfigState.State, AckState):

    def __init__(self, app_name, input_dir: str = "/mnt/input", output_dir: str = "/mnt/output"):
        ConfigState.State.__init__(self, app_name, input_dir, output_dir)
        self.design_df = None
        self.counts_df = None

    def run(self) -> str or None:
        self.lazy_init()
        self.read_config()
        self.finalize_config()
        self.store('config', self.config)
        self.store('smpc_used', self.config.get('use_smpc', False))
        self.read()

        # send the list of features and cohort names to the server
        data_to_send = [self.load('local_sample_count')]
        self.log(f"**** Data: {data_to_send}")
        if self.load('smpc_used'):
            if hasattr(data_to_send, "tolist"):
                data_to_send = data_to_send.tolist()

        self.log(f"**** Data: {data_to_send}")
        self.instant_aggregate(name='global_sample_count', data=data_to_send, use_smpc=self.load('smpc_used'))
        self.send_data_to_coordinator(data=self.load('local_features'), use_smpc=False)

    def read(self):
        counts_df, design_df = readfiles(self.load('input_files')['counts'][0],
                                         self.load('input_files')['design'][0])

        g1 = sorted(list(filter(None, self.config['group1'].split(","))))
        g2 = sorted(list(filter(None, self.config['group2'].split(","))))
        conf = sorted(filter(None, list(self.config['confounders'].split(','))))
        loc_f = sorted(counts_df.index.values)
        samples = sorted(list(set(counts_df.columns.values).intersection(design_df.index.values)))
        design_cols = set(design_df.columns)
        if len(set(g1).intersection(design_cols)) == 0:
            self.log("\tClass labels %s are missing in the design matrix." % ",".join(g1))
            self.update(state=op_state.ERROR)
        if len(set(g2).intersection(design_cols)) == 0:
            self.log("\t Class labels %s are missing in the design matrix." % ",".join(g2))
            self.update(state=op_state.ERROR)
        missing_conf_variables = set(conf).difference(set(design_cols))
        if len(missing_conf_variables) > 0:
            self.log(
                "\tConfounder variable(s) are missing in the design matrix: %s." % ",".join(missing_conf_variables))
            self.update(state=op_state.ERROR)
        counts_df = counts_df.loc[loc_f, samples]
        var = g1 + g2 + conf
        design_df = design_df.loc[samples, var]
        self.store('group1', g1)
        self.store('group2', g2)
        self.store('confounders', conf)
        self.store('local_features', loc_f)
        self.store('samples', samples)
        self.store('local_sample_count', len(samples))
        self.store('counts_df', counts_df)
        self.log("\tCount matrix: %s features x %s samples" % counts_df.shape)
        self.store('variables', var)
        self.store('design_df', design_df)
        # name cohort
        self.store('cohort_name', "Cohort_" + self.id)
        self.store('norm_factors', np.ones(counts_df.shape[1]))


class GlobalMean(AppState):
    def __init__(self):
        super().__init__()
        self.gene_name_list = None
        self.cohort_effects = None
        self.global_sample_count = None

    def register(self):
        self.register_transition('CPM_Cut_Off', Role.COORDINATOR)

    def run(self) -> str or None:

        self.global_sample_count = (np.array(self.load('global_sample_count'))).tolist()
        clients_features = self.gather_data()
        self.aggregate_features(clients_features)
        self.log_data()
        self.store('gene_name_list', self.gene_name_list)
        self.store('n_features', len(self.gene_name_list))
        self.store('cohort_effects', self.cohort_effects)

        self.broadcast_data(data=[self.gene_name_list, self.cohort_effects])
        self.store("server_vars", self.load('group1') + self.load('group2'))

    def aggregate_features(self, clients_features):
        feature_lists = []
        for client_data in clients_features:
            feature_lists.append(client_data)
        self.store('cohort_names', [f"Cohort_{i}" for i in range(len(self.clients))])
        self.cohort_effects = {f"Cohort_{i}": id for i, id in enumerate(self.clients) if id != self.id}

        shared_features = set(feature_lists[0])
        for feature_list in feature_lists[1:]:
            shared_features = shared_features.intersection(set(feature_list))
        self.gene_name_list = sorted(list(shared_features))

    def log_data(self):
        self.log(f"#############\n"
                 f"Total samples: {self.global_sample_count}\n"
                 f"Shared features: {self.gene_name_list[:3]}, ...,"
                 f" {len(self.gene_name_list)} features\n"
                 f"Joined cohorts: {self.load('cohort_names')} \n"
                 f"Cohort effects added: {self.cohort_effects}\n"
                 f"############")
