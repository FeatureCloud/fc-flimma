

large_n = 10
min_prop = 0.7
tol = 1e-14


def get_k_n(variables, confounders, cohort_names, gene_name_list):
    k = len(variables) + len(confounders) + len(cohort_names) - 1
    n = len(gene_name_list)
    return k, n


