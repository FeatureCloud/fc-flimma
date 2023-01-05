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

from FeatureCloud.app.engine.app import AppState
import pandas as pd
import numpy as np
from scipy import linalg
import statsmodels.api as sm
from scipy.special import digamma, polygamma
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from copy import copy
from States import get_k_n
from CustomStates.AckState import AckState


class SSE(AckState):
    def __init__(self):
        self.weighted = False
        self.beta = None

    def run(self) -> str or None:
        self.beta = self.await_data()
        self.compute_sse_step_parameters()
        log_count, log_count_conversion_term = self.mean_log_count_step()
        data_to_send = {'sum_sample_count': self.load('local_sample_count'),
                        'sum_sse': self.load('sse'),
                        'sum_cov':self.load('cov_coefficient'),
                        'sum_log_count': log_count,
                        'sum_log_count_conversion': log_count_conversion_term.item()}
        for name, data in data_to_send.items():
            self.instant_aggregate(name=name, data=data, use_smpc=self.load('smpc_used'))

    def compute_sse_step_parameters(self):
        x_matrix = self.load('design_df').values
        y_matrix = self.load('log_cpm').values
        n = y_matrix.shape[0]
        sse = np.zeros(n)
        if self.weighted:
            w_square = np.sqrt(self.load('weight'))
            y_matrix = np.multiply(y_matrix, w_square)

        mu = self.load('mu')
        for i in range(0, n):  #
            y = y_matrix[i, :]
            if self.weighted:
                x_w = np.multiply(x_matrix, w_square[i, :].reshape(-1, 1))
                mu[i,] = x_w @ self.beta[i, :]  # fitted logCPM

            else:
                mu[i,] = x_matrix @ self.beta[i, :]  # fitted logCPM

            sse[i] = np.sum((y - mu[i,]) ** 2)  # local SSE
        self.store('mu', mu)
        self.store('sse', sse)
        Q, R = np.linalg.qr(x_matrix)
        self.store('cov_coefficient', R.T @ R)

    def mean_log_count_step(self):
        # Exact procedure will be repeated two times! (weighted = {True, False})
        # Get nothing from the server!
        log_count = self.load('log_cpm').sum(axis=1).values

        log_count_conversion_term = np.sum(np.log2(self.load('lib_sizes') + 1))
        return log_count, log_count_conversion_term


class AggregateSSE(AppState):
    def __init__(self):
        self.weighted = False
        self.results = {}

    def run(self) -> str or None:

        self.aggregate_sse()
        self.aggregate_mean_log_count()

        if self.weighted:
            self.broadcast_data(self.py_lowess)
            # self.weighted = not self.weighted
        else:
            self.ebayes_step()
            self.table["GENE"] = self.table.index
            self.table.rename(columns={'logFC': "EFFECTSIZE", 'adj.P.Val': "P"}).to_csv("/mnt/output/tabel.csv", sep=",")
            # self.table.to_csv("/mnt/output/tabel.csv", sep=",")
            data_to_send = [self.table['logFC'].values, self.table['adj.P.Val'].values, self.table.index.values]
            self.store('effectsize', data_to_send[0])
            self.store('P', data_to_send[1])
            self.store('gene', data_to_send[2])
            self.log('Broadcasting the data')
            self.broadcast_data(data=data_to_send)

    def aggregate_sse(self):
        self.global_sample_count = np.array(self.load('sum_sample_count')) / len(self.clients)
        self.total_cov = self.load('sum_cov') / len(self.clients)
        self.sse = self.load('sum_sse') / len(self.clients)
        self.cov_coefficient = linalg.inv(self.total_cov)
        k, n = get_k_n(self.load('server_vars'), self.load('confounders'), self.load('cohort_names'),
                       self.load('gene_name_list'))
        # estimated residual variance
        self.variance = self.sse / (self.global_sample_count - k)

        # estimated residual standard deviations
        self.sigma = np.sqrt(self.variance)

        # degrees of freedom
        self.degree_of_freedom = np.ones(n) * (self.global_sample_count - k)

    def aggregate_mean_log_count(self):
        self.total_log_count = self.load('sum_log_count') / len(self.clients)
        self.total_log_count_conversion = self.load('sum_log_count_conversion') / len(self.clients)
        self.total_log_count_conversion = self.total_log_count_conversion / self.global_sample_count - 6 * np.log2(10)
        self.mean_count = self.total_log_count / self.global_sample_count
        self.mean_log_count = self.mean_count + self.total_log_count_conversion

        self.delta = (max(self.mean_log_count) - min(self.mean_log_count)) * 0.01
        self.lowess = sm.nonparametric.lowess
        self.py_lowess = self.lowess(self.sigma ** 0.5,
                                     self.mean_log_count,
                                     frac=0.5,
                                     delta=self.delta,
                                     return_sorted=True, is_sorted=False)

    def get_k_n(self):
        k = len(self.load('variables')) + len(self.load('confounders')) + len(self.load('cohort_names')) - 1
        n = len(self.load('gene_name_list'))
        return k, n

    def ebayes_step(self):
        self.log("making contrasts ...")
        self.make_contrasts(contrast_list=[([self.load('group1')[0]], [self.load('group2')[0]])])
        self.log("contrast matrix:")
        self.log(self.contrast_matrix)
        self.log("Fitting contrasts ...")
        self.fit_contrasts()

        self.log("empirical Bayes ...")
        self.e_bayes()

    def make_contrasts(self, contrast_list):
        """Creates contrast matrix given design matrix and pairs or columns to compare.
        For example:
        contrasts = [([A],[B]),([A,B],[C,D])] defines two contrasts:\n
        A-B and (A and B) - (C and D)."""
        df = {}
        # conditions = self.variables + self.confounders + self.cohort_names[0:-1]
        conditions = self.load('server_vars') + self.load('confounders') + self.load('cohort_names')[0:-1]
        for contrast in contrast_list:
            group1, group2 = contrast
            for name in group1 + group2:
                if name not in conditions:
                    self.log(name, "not found in the design matrix.")
                    exit(1)
            contrast_name = "".join(map(str, group1)) + "_vs_" + "".join(map(str, group2))
            series = pd.Series(data=np.zeros(len(conditions)), index=conditions)
            series[group1] = 1
            series[group2] = -1
            df[contrast_name] = series

        self.contrast_matrix = pd.DataFrame.from_dict(df).values

    def fit_contrasts(self):
        n_coef = self.cov_coefficient.shape[1]
        #	Correlation matrix of estimable coefficients
        #	Test whether design was orthogonal
        if not np.any(self.cov_coefficient):
            self.log("no coefficient correlation matrix found in fit - assuming orthogonal")
            correlation_matrix = np.identity(n_coef)
            orthog = True
        else:
            self.log("coefficient correlation matrix is found")
            correlation_matrix = self.cov2cor()
            self.log("cov2cor() is called")
            if correlation_matrix.shape[0] * correlation_matrix.shape[1] < 2:
                orthog = True
            else:
                if np.sum(np.abs(np.tril(correlation_matrix, k=-1))) < 1e-12:
                    orthog = True
                else:
                    orthog = False

        #	Replace NA coefficients with large (but finite) standard deviations
        #	to allow zero contrast entries to clobber NA coefficients.
        self.std_unscaled = self.load('std_unscaled')
        if np.any(np.isnan(self.load('beta'))):
            self.log("Replace NA coefficients with large (but finite) standard deviations")
            np.nan_to_num(self.load('beta'), nan=0)
            np.nan_to_num(self.std_unscaled, nan=1e30)

        self.store('beta', self.load('beta').dot(self.contrast_matrix))
        # New covariance coefficiets matrix
        self.cov_coefficient = self.contrast_matrix.T.dot(self.cov_coefficient).dot(self.contrast_matrix)
        if orthog:
            self.std_unscaled = np.sqrt((self.std_unscaled ** 2).dot(self.contrast_matrix ** 2))
        else:
            n_genes = self.load('beta').shape[0]
            U = np.ones((n_genes, self.contrast_matrix.shape[1]))  # genes x contrasts
            o = np.ones(n_coef)
            R = np.linalg.cholesky(correlation_matrix).T
            for i in range(0, n_genes):
                RUC = R @ (self.std_unscaled[i,] * self.contrast_matrix.T).T
                U[i,] = np.sqrt(o @ RUC ** 2)
            self.std_unscaled = U

    def cov2cor(self):
        cor = np.diag(self.cov_coefficient) ** -0.5 * self.cov_coefficient
        cor = cor.T * np.diag(self.cov_coefficient) ** -0.5
        np.fill_diagonal(cor, 1)
        return cor

    # ######## eBayes
    def trigamma(self, x):
        return polygamma(1, x)

    def psigamma(self, x, deriv=2):
        return polygamma(deriv, x)

    def trigammaInverse(self, x):
        if not hasattr(x, '__iter__'):
            x_ = np.array([x])
        for i in range(0, x_.shape[0]):
            if np.isnan(x_[i]):
                x_[i] = np.NaN
            elif x > 1e7:
                x_[i] = 1. / np.sqrt(x[i])
            elif x < 1e-6:
                x_[i] = 1. / x[i]
        # Newton's method
        y = 0.5 + 1.0 / x_
        for i in range(0, 50):
            tri = self.trigamma(y)
            dif = tri * (1.0 - tri / x_) / self.psigamma(y, deriv=2)
            y = y + dif
            if np.max(-dif / y) < 1e-8:  # tolerance
                return y

        self.log("Warning: Iteration limit exceeded")
        return y

    def moderatedT(self, covariate=False, robust=False, winsor_tail_p=(0.05, 0.1)):

        # var,df_residual,coefficients,stdev_unscaled,
        self.squeeze_var(covariate=covariate, robust=robust, winsor_tail_p=winsor_tail_p)

        self.results["s2_prior"] = self.results["var_prior"]
        self.results["s2_post"] = self.results["var_post"]
        del self.results["var_prior"]
        del self.results["var_post"]

        self.results["t"] = self.load('beta') / self.std_unscaled
        self.results["t"] = self.results["t"].T / np.sqrt(self.results["s2_post"])
        self.df_total = self.degree_of_freedom + self.results["df_prior"]
        df_pooled = sum(self.degree_of_freedom)
        self.df_total = np.minimum(self.df_total, df_pooled)  # component-wise min

        self.results["p_value"] = 2 * t.cdf(-np.abs(self.results["t"]), df=self.df_total)
        self.results["p_value"] = self.results["p_value"].T
        self.results["t"] = self.results["t"].T

    def squeeze_var(self, covariate=False, robust=False, winsor_tail_p=(0.05, 0.1)):
        '''Estimates df and var priors and computes posterior variances.'''
        if robust:
            # TBD fitFDistRobustly()
            self.log("Set robust=False.")
            return
        else:
            var_prior, df_prior = self.fitFDist(covariate=covariate)

        if np.isnan(df_prior):
            self.log("Error: Could not estimate prior df")
            return

        var_post = self.posterior_var(var_prior=var_prior, df_prior=df_prior)
        self.results = {"df_prior": df_prior, "var_prior": var_prior, "var_post": var_post}

    def fitFDist(self, covariate=False):
        '''Given x (sigma^2) and df1 (degree_of_freedom),
        fits x ~ scale * F(df1,df2) and returns
        estimated df2 and scale (s0^2)'''

        if covariate:
            # TBD
            self.log("Set covariate=False.")
            return

        # Avoid zero variances
        variances = [max(var, 0) for var in self.variance]
        median = np.median(variances)
        if median == 0:
            self.log("Warning: More than half of residual variances are exactly zero: eBayes unreliable")
            median = 1
        else:
            if 0 in variances:
                self.log("Warning: Zero sample variances detected, have been offset (+1e-5) away from zero")

        variances = [max(var, 1e-5 * median) for var in variances]
        z = np.log(variances)
        e = z - digamma(self.degree_of_freedom * 1.0 / 2) + np.log(self.degree_of_freedom * 1.0 / 2)
        emean = np.nanmean(e)
        evar = np.nansum((e - emean) ** 2) / (len(variances) - 1)

        # Estimate scale and df2
        evar = evar - np.nanmean(self.trigamma(self.degree_of_freedom * 1.0 / 2))
        if evar > 0:
            df2 = 2 * self.trigammaInverse(evar)
            s20 = np.exp(emean + digamma(df2 * 1.0 / 2) - np.log(df2 * 1.0 / 2))
        else:
            df2 = np.Inf
            s20 = np.exp(emean)

        return s20, df2

    def posterior_var(self, var_prior=np.ndarray([]), df_prior=np.ndarray([])):
        '''.squeezeVar()'''
        var = self.variance
        df = self.degree_of_freedom
        ndxs = np.argwhere(np.isfinite(var)).reshape(-1)
        # if not infinit vars
        if len(ndxs) == len(var):  # np.isinf(df_prior).any():
            return (df * var + df_prior * var_prior) / (df + df_prior)  # var_post
        # For infinite df.prior, set var_post = var_prior
        var_post = np.repeat(var_prior, len(var))
        for ndx in ndxs:
            var_post[ndx] = (df[ndx] * var[ndx] + df_prior * var_prior) / (df[ndx] + df_prior)
        return var_post

    def tmixture_matrix(self, var_prior_lim=False, proportion=0.01):
        tstat = self.results["t"]
        std_unscaled = self.std_unscaled
        df_total = self.df_total
        ncoef = self.results["t"].shape[1]
        v0 = np.zeros(ncoef)
        for j in range(0, ncoef):
            v0[j] = self.tmixture_vector(tstat[:, j], std_unscaled[:, j], df_total, proportion, var_prior_lim)
        return v0

    def tmixture_vector(self, tstat, std_unscaled, df, proportion, var_prior_lim):
        ngenes = len(tstat)

        # Remove missing values
        notnan_ndx = np.where(~np.isnan(tstat))[0]
        if len(notnan_ndx) < ngenes:
            tstat = tstat[notnan_ndx]
            std_unscaled = std_unscaled[notnan_ndx]
            df = df[notnan_ndx]

        # ntarget t-statistics will be used for estimation

        ntarget = int(np.ceil(proportion / 2 * ngenes))
        if ntarget < 1:  #
            return

            # If ntarget is v small, ensure p at least matches selected proportion
        # This ensures ptarget < 1
        p = np.maximum(ntarget * 1.0 / ngenes, proportion)

        # Method requires that df be equal
        tstat = abs(tstat)
        MaxDF = np.max(df)
        i = np.where(df < MaxDF)[0]
        if len(i) > 0:
            TailP = t.logcdf(tstat[i], df=df[i])
            # PT: CDF of t-distribution: pt(tstat[i],df=df[i],lower.tail=FALSE,log.p=TRUE)
            # QT - qunatile funciton - returns a threshold value x
            # below which random draws from the given CDF would fall p percent of the time. [wiki]
            tstat[i] = t.ppf(np.exp(TailP), df=MaxDF)  # QT: qt(TailP,df=MaxDF,lower.tail=FALSE,log.p=TRUE)
            df[i] = MaxDF

        # Select top statistics
        order = tstat.argsort()[::-1][:ntarget]  # TBD: ensure the order is decreasing
        tstat = tstat[order]
        v1 = std_unscaled[order] ** 2

        # Compare to order statistics
        rank = np.array(range(1, ntarget + 1))
        p0 = 2 * t.sf(tstat, df=MaxDF)  # PT
        ptarget = ((rank - 0.5) / ngenes - (1.0 - p) * p0) / p
        v0 = np.zeros(ntarget)
        pos = np.where(ptarget > p0)[0]
        if len(pos) > 0:
            qtarget = -t.ppf(ptarget[pos] / 2, df=MaxDF)  # qt(ptarget[pos]/2,df=MaxDF,lower.tail=FALSE)
            v0[pos] = v1[pos] * ((tstat[pos] / qtarget) ** 2 - 1)

        if var_prior_lim[0] and var_prior_lim[1]:
            v0 = np.minimum(np.maximum(v0, var_prior_lim[0]), var_prior_lim[1])

        return np.mean(v0)

    def b_stat(self, std_coef_lim=np.array([0.1, 4]), proportion=0.01):
        var_prior_lim = std_coef_lim ** 2 / np.median(self.results["s2_prior"])
        # print("Limits for var.prior:",var_prior_lim)

        self.results["var_prior"] = self.tmixture_matrix(proportion=0.01, var_prior_lim=var_prior_lim)

        nan_ndx = np.argwhere(np.isnan(self.results["var_prior"]))
        if len(nan_ndx) > 0:
            if self.results["var.prior"][nan_ndx] < - 1.0 / self.results["s2_prior"]:
                self.log("Warning: Estimation of var.prior failed - set to default value")
        r = np.outer(np.ones(self.results["t"].shape[0]), self.results["var_prior"])
        r = (self.std_unscaled ** 2 + r) / self.std_unscaled ** 2
        t2 = self.results["t"] ** 2

        valid_df_ndx = np.where(self.results["df_prior"] <= 1e6)[0]
        if len(valid_df_ndx) < len(self.results["df_prior"]):
            self.log("Large (>1e6) priors for DF:", len(valid_df_ndx))
            kernel = t2 * (1 - 1.0 / r) / 2
            for i in valid_df_ndx:
                kernel[i] = (1 + self.df_total[i]) / 2 * np.log(
                    (t2[i, :].T + self.df_total[i]) / ((t2[i, :] / r[i, :]).T + self.df_total[i]))
        else:
            kernel = (1 + self.df_total) / 2 * np.log((t2.T + self.df_total) / ((t2 / r).T + self.df_total))

        self.results["lods"] = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel.T

    def top_table_t(self, adjust="fdr_bh", p_value=1.0, lfc=0, confint=0.95):
        feature_names = self.load('gene_name_list')
        self.results["logFC"] = pd.Series(self.load('beta')[:, 0], index=feature_names)

        # confidence intervals for LogFC
        if confint:
            alpha = (1.0 + confint) / 2
            margin_error = np.sqrt(self.results["s2_post"]) * self.std_unscaled[:, 0] * t.ppf(alpha,
                                                                                              df=self.df_total)
            self.results["CI.L"] = self.results["logFC"] - margin_error
            self.results["CI.R"] = self.results["logFC"] + margin_error
        # adjusting p-value for multiple testing
        if_passed, adj_pval, alphacSidak, alphacBonf = multipletests(self.results["p_value"][:, 0], alpha=p_value,
                                                                     method=adjust,
                                                                     is_sorted=False, returnsorted=False)
        self.results["adj.P.Val"] = pd.Series(adj_pval, index=feature_names)
        self.results["P.Value"] = pd.Series(self.results["p_value"][:, 0], index=feature_names)
        # make table
        self.table = copy(self.results)
        # remove 'df_prior', 's2_prior', 's2_post', 'df_total','var_prior'
        for key in ['df_prior', 's2_prior', 's2_post', 'var_prior', "p_value"]:
            del self.table[key]
        self.table["t"] = pd.Series(self.table["t"][:, 0], index=feature_names)
        self.table["lods"] = pd.Series(self.table["lods"][:, 0], index=feature_names)
        self.table = pd.DataFrame.from_dict(self.table)

    def e_bayes(self):
        covariate = False  # Amean for limma-trend
        robust = False  #
        winsor_tail_p = (0.05, 0.1)  # needed for fitFDistRobustly()

        self.moderatedT(covariate=covariate, robust=robust, winsor_tail_p=winsor_tail_p)
        self.results["AveExpr"] = self.mean_count

        self.b_stat(std_coef_lim=np.array([0.1, 4]), proportion=0.01)

        self.top_table_t(adjust="fdr_bh", p_value=1.0, lfc=0, confint=0.95)
        self.table = self.table.sort_values(by="P.Value")


class WriteResults(AppState):
    def run(self) -> str or None:
        self.update(message="Writing results...")
        if self.is_coordinator:
            effect_size = self.load('effectsize')
            p_values = self.load('P')
            genes = self.load('gene')
        else:
            effect_size, p_values, genes = self.await_data()
        df = pd.DataFrame(data={'EFFECTSIZE': effect_size, 'P': p_values, 'GENE': genes},
                          columns=['EFFECTSIZE', 'P', 'GENE'], index=None)
        df['SNP'] = None
        self.log(f"Write the results as {self.load('output_files')['table'][0]}")
        df.to_csv(self.load('output_files')['table'][0], sep=",", index=False)
