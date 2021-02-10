import math
import numpy as np
from scipy.special import iv
from scipy.interpolate import interp1d
from spike_statistics.core import block_bootstrap
import pandas as pd
import scipy.stats


class VonMisesKDE():
    def __init__(self, data, weights=[], kappa=1.0):
        """ Constructor. Define KDE options and input data
        
        @param List[float] Input data points in radians. Normalized by method to range [0, 2*pi]
        @param List[float] Optional. Weights for input samples
        @param float kappa Optional. Kappa parameter of Von Mises pdf
        
        Acknowledgement:
        Copied from https://github.com/engelen/vonmiseskde
        """
        
        # Input data
        self.data = self.normalizeAngles(np.asarray(data))
        
        # Input data weights
        self.weights = np.asarray(weights)
        self.weights = self.weights / np.sum(weights)
        
        # Model parameter
        self.kappa = kappa
        
        # Generate KDE
        self.kde()
    
    def normalizeAngles(self, data):
        """ Normalize a list of angles (in radians) to the range [-pi, pi]
        
        @param List[float] Input angles (in radians)
        @return List[float] Normalized angles (in radians)
        """
        # Change range to 0 to 2 pi
        data = np.array(data % (np.pi * 2))

        # Change range to -1 pi to 1 pi
        data[data > np.pi] = data[data > np.pi] - np.pi * 2
        
        return data
        
    def vonMisesPDF(self, alpha, mu=0.0):
        """ Probability density function of Von Mises pdf for input points
        
        @param List[float] alpha List-like or single value of input values
        @return List[float] List of probabilities for input points
        """
        
        num = np.exp(self.kappa * np.cos(alpha - mu))
        den = 2 * np.pi * iv(0, self.kappa)

        return num / den

    def kde(self):
        """ Calculate kernel density estimator distribution function """
        
        plot = True
        
        # Input data
        x_data = np.linspace(-math.pi, math.pi, 1000)

        # Kernels, centered at input data points
        kernels = []

        for datapoint in self.data:
            # Make the basis function as a von mises PDF
            kernel = self.vonMisesPDF(x_data, mu=datapoint)
            kernels.append(kernel)
    
        # Handle weights
        if len(self.weights) > 0:
            kernels = np.asarray(kernels)
            weighted_kernels = np.multiply(kernels, self.weights[:, None])
        else:
            weighted_kernels = kernels
        
        # Normalize pdf
        vmkde = np.sum(weighted_kernels, axis=0)
        vmkde = vmkde / np.trapz(vmkde, x=x_data)

        self.fn = interp1d(x_data, vmkde)
    
    def evaluate(self, input_x):
        """ Evaluate the KDE at some inputs points
        
        @param List[float] input_x Input points
        @param List[float] Probability densities at input points
        """
        
        # Normalize inputs
        input_x = self.normalizeAngles(input_x)

        return self.fn(input_x)
    
    
def compute_weighted_mean_sem(data, label, groupby='entity'):
    group = data.groupby(groupby)
    tmp = [d.loc[:, label].dropna().values for _, d in group]
    values = np.concatenate(tmp)
    if len(values) == 0:
        return [np.nan] * 3
    weights = np.concatenate([np.ones_like(a) / len(a) for a in tmp])
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average)**2, weights=weights)
    sem = np.sqrt(variance / len(values))
    return average, sem, len(values)


def compute_confidence_interval(data, alpha=0.05):
    stat = np.sort(data.dropna())
    n = len(stat)
    if n == 0:
        return np.nan, np.nan
    low = stat[int((alpha / 2.0) * n)]
    high = stat[int((1 - alpha / 2.0) * n)]
#     low, high = np.percentile(data.dropna(), [(alpha / 2.0) * 100, (1 - alpha / 2.0) * 100])
    return low, high


def pvalue(df, df_bootstrap, control_key, case_key):
    '''
    pvalue from bootstrap results, shifts the bootstrapped distribution
    '''
    case, b = df_bootstrap[case_key].dropna(), df_bootstrap[control_key].dropna()
    if len(case) == 0 or len(b) == 0:
        return [np.nan] * 3
    
    n = len(case)
    
    average_case, _, _ = compute_weighted_mean_sem(df, case_key)
    average_control, _, _ = compute_weighted_mean_sem(df, control_key)
    
    low, high = compute_confidence_interval(average_control - case)
    
    case_shift = case - case.mean()
    diff = abs(average_case - average_control)    
    
    pval = (np.sum(case_shift > diff) + np.sum(case_shift < - diff)) / n
    
    return pval, low, high


def bootstrap_pvalue(df, control_key, case_key, n_boots=100, n_samples=10, n_blocks=4, delta=0.0, alpha=0.05):
    '''
    pvalue from bootstrap results, shifts the sample distribution
    '''
    average_case, _, _ = compute_weighted_mean_sem(df, case_key)
    average_control, _, _ = compute_weighted_mean_sem(df, control_key)
    group = df.groupby('entity')
    case_values = np.array([d.loc[:, case_key].dropna().values for _, d in group if d.loc[:, case_key].count() > 0])
    control_values = np.array([d.loc[:, control_key].dropna().values for _, d in group if d.loc[:, control_key].count() > 0])
    
    case_boot = block_bootstrap(case_values - average_case + delta, n_boots=n_boots, n_samples=n_samples, n_blocks=n_blocks, statistic=lambda x:np.ravel(x))
    control_boot = block_bootstrap(control_values - average_control, n_boots=n_boots, n_samples=n_samples, n_blocks=n_blocks, statistic=lambda x:np.ravel(x))
    
    observed_diff = abs(average_case - average_control)
    
    print(case_boot.shape)
    
    # direct
    diffs = case_boot.mean(1) - control_boot.mean(1)
    low, high = np.percentile(diffs, [(alpha / 2.0) * 100, (1 - alpha / 2.0) * 100])
    pval = (np.sum(diffs > observed_diff) + np.sum(diffs < - observed_diff)) / len(case_boot)
    
    # tstat
#     T = [scipy.stats.ttest_ind(a, b).statistic for a, b in zip(case_boot, control_boot)]
#     pval_t = (1 + np.sum(np.abs(T) > np.abs(scipy.stats.ttest_ind(df.loc[:, case_key].dropna(), df.loc[:, control_key].dropna()).statistic))) / (len(case_boot)+1)

    low_, high_ = np.percentile(average_control - case_boot.mean(1) + average_case, [(alpha / 2.0) * 100, (1 - alpha / 2.0) * 100])
    
    pval_ = (np.sum(case_boot.mean(1) > observed_diff) + np.sum(case_boot.mean(1) < - observed_diff)) / len(case_boot)
    
    return pval, low, high, pval_, low_, high_ 


def power(df, control_key, case_key, delta, pval=0.05, n_boots=100, n_samples=10, n_blocks=4, ntest=100):
    
    pvalues = []
    for _ in range(ntest):
        p = bootstrap_pvalue(
            df=df, 
            control_key=control_key, 
            case_key=case_key, 
            n_boots=n_boots, 
            n_samples=n_samples, 
            n_blocks=n_blocks, 
            delta=delta
        )
        pvalues.append(p)
    return pvalues
#     return np.mean(np.array(pvalues) < pval)


# def wilcoxon_power(df, control_key, case_key, delta, pval=0.05, n_boots=100, n_samples=None, n_blocks=4, ntest=100):
    
#     pvalues = []
#     for _ in range(ntest):
#         p = bootstrap_pvalue(
#             df=df, 
#             control_key=control_key, 
#             case_key=case_key, 
#             n_boots=n_boots, 
#             n_samples=n_samples, 
#             n_blocks=n_blocks, 
#             delta=delta
#         )
#         pvalues.append(p)
#     return pvalues
#     return np.mean(np.array(pvalues) < pval)



def rename(name):
    return name.replace("_field", "-field").replace("_", " ").capitalize()


def make_bootstrap_table(results, bootstrapped_results, labels):
    stat_formatted = pd.DataFrame()
    stat_values = pd.DataFrame()
    
    for key, df in bootstrapped_results.items():
        Key = rename(key)

        for label in labels:
            average, sem, n = compute_weighted_mean_sem(results[key], label)
            if np.isnan(average):
                stat_formatted.loc[label, Key] = np.nan
                stat_values.loc[label, key] = np.nan
            else:
                stat_formatted.loc[label, Key] = "{:.1e} ± {:.1e} ({})".format(average, sem, n)
                stat_values.loc[label, Key] = average

        for i, c1 in enumerate(df.columns):
            for c2 in df.columns[i+1:]:
                pval, low, high = pvalue(results[key], df, c1, c2)
                if np.isnan(pval):
                    stat_formatted.loc[f'{c1} - {c2}', Key] = np.nan
                    stat_values.loc[f'{c1} - {c2}', key] = np.nan
                else:
                    stat_formatted.loc[f'{c1} - {c2}', Key] = "{:.1e} [{:.1e}, {:.1e}]".format(pval, low, high)
                    stat_values.loc[f'{c1} - {c2}', key] = pval
                
    return stat_formatted, stat_values


def bootstrap_results(results, labels, n_boots=100, n_samples=10, n_blocks=4):
    bootstrapped_results = {}
    for key, df in results.items():
        bootstrapped_results[key] = pd.DataFrame()
        group = df.groupby('entity')
        for label in labels:
            entity_values = np.array([d.loc[:, label].dropna().values for _, d in group if d.loc[:, label].count() > 0])
            if len([i for j in entity_values for i in j]) < 3: # less than total 3 samples
                    boot_samples = np.ones(n_boots) * np.nan
            else:
                boot_samples = block_bootstrap(entity_values, n_boots=n_boots, n_samples=n_samples, n_blocks=n_blocks, statistic=np.mean)
            bootstrapped_results[key].loc[:, label] = np.ravel(boot_samples)
    return bootstrapped_results


def wilcoxon(df, keys):
    dff = df.loc[:,[keys[0], keys[1]]].dropna()
    if dff.empty:
        return [np.nan] * 3
    statistic, pvalue = scipy.stats.wilcoxon(
        dff[keys[0]], 
        dff[keys[1]],
        alternative='two-sided')

    return statistic, pvalue, len(dff)


def MWU(df, keys):
    '''
    Mann Whitney U
    '''
    d1 = df[keys[0]].dropna()
    d2 = df[keys[1]].dropna()
    Uvalue, pvalue = scipy.stats.mannwhitneyu(
        d1, d2, alternative='two-sided')

    return Uvalue, pvalue

    
def normality(df, key):
    statistic, pvalue = scipy.stats.normaltest(
        df[key].dropna())

    return statistic, pvalue


def make_paired_table(results, labels):
    paired_stat = pd.DataFrame()
    paired_stat_values = pd.DataFrame()
    for key, df in results.items():
        Key = rename(key)

        for label in labels:
            if df[label].count() < 8:
                paired_stat.loc[f'Normality {label}', Key] = np.nan
            else:
                statistic, pval = normality(df, label)
                if np.isnan(statistic):
                    paired_stat.loc[f'Normality {label}', Key] = np.nan
                else:
                    paired_stat.loc[f'Normality {label}', Key] = "{:.1e}, {:.1e}".format(statistic, pval)

        for i, c1 in enumerate(labels):
            for c2 in labels[i+1:]:
                statistic, pval, n = wilcoxon(df, [c1, c2])
                if np.isnan(statistic):
                    paired_stat.loc[f'Wilcoxon {c1} - {c2}', Key] = np.nan
                else:
                    paired_stat.loc[f'Wilcoxon {c1} - {c2}', Key] = "{:.1e}, {:.1e}, ({})".format(statistic, pval, n)
                paired_stat_values.loc[f'{c1} - {c2}', key] = pval
    return paired_stat, paired_stat_values