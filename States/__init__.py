# from pyrecord import Record
# from FeatureCloud.app.engine.app import Role

large_n = 10
min_prop = 0.7
tol = 1e-14


def get_k_n(variables, confounders, cohort_names, gene_name_list):
    print(variables, confounders, cohort_names)
    k = len(variables) + len(confounders) + len(cohort_names) - 1
    n = len(gene_name_list)
    return k, n


# Transition = Record.create_type('target', 'role', 'name')
# State = Record.create_type('name', 'role', 'transition')
# States = Record.create_type('state', 'transition')
# State(target='', role= Role.BOTH, name='initial')
# Transition(self.register_transition('Global_Mean', Role.COORDINATOR)
#         self.register_transition('CPM_Cut_Off', Role.PARTICIPANT))
#
# states =
#     State()
#
# ]