import math
import numpy as np
from scipy.special import iv
from scipy.interpolate import interp1d
from spike_statistics.core import block_bootstrap
import pandas as pd
import scipy.stats
from functools import reduce

import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
# from statsmodels.tools.sm_exceptions import ConvergenceWarning
# warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore')

from tqdm.notebook import tqdm
tqdm.pandas()


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
    
    
def get_queries(stim_location='stim_location=="ms"'):
    
    colors = ['#1b9e77','#d95f02','#7570b3','#e7298a']
    labels = ['Baseline I', '11 Hz', 'Baseline II', '30 Hz']
    # Hz11 means that the baseline session was indeed before an 11 Hz session
    queries = [
        'baseline and i and Hz11', 
        f'frequency==11 and {stim_location}', 
        'baseline and ii and Hz30',
        f'frequency==30 and {stim_location}'
    ]
    gridcell_query = (
        'gridness > gridness_threshold and '
        'information_rate > information_rate_threshold and '
        'gridness > .2 and '
        'average_rate < 25'
    )
    return queries, labels, colors, gridcell_query
    
    
def load_data_frames(queries=None, labels=None, colors=None, stim_location='stim_location=="ms"'):
    cell_types = [
        'ns_inhibited', 
        'ns_not_inhibited',
        'gridcell',
        'bs_not_gridcell'
    ]
    import expipe
    import septum_mec.analysis.data_processing as dp
    if queries is None:
        queries, labels, colors, gridcell_query = get_queries(stim_location)
    else:
        _, _, _, gridcell_query = get_queries()
        
    label_nums = list(range(len(labels)))
    
    project_path = dp.project_path()
    project = expipe.get_project(project_path)
    actions = project.actions
    
    
    identification_action = actions['identify-neurons']
    sessions = pd.read_csv(identification_action.data_path('sessions'))
    units = pd.read_csv(identification_action.data_path('units'))
    session_units = pd.merge(sessions, units, on='action')
    
    statistics_action = actions['calculate-statistics']
    statistics_results = pd.read_csv(statistics_action.data_path('results'))
    statistics = pd.merge(session_units, statistics_results, how='left')
    statistics['unit_day'] = statistics.apply(lambda x: str(x.unit_idnum) + '_' + x.action.split('-')[1], axis=1)
    
#     statistics_action_extra = actions['calculate-statistics-extra']
#     statistics_action_extra = pd.read_csv(statistics_action_extra.data_path('results'))
#     statistics = pd.merge(statistics, statistics_action_extra, how='left')
    
    stim_response_action = actions['stimulus-response']
    stim_response_results = pd.read_csv(stim_response_action.data_path('results'))
    statistics = pd.merge(statistics, stim_response_results, how='left')
    
    shuffling = actions['shuffling']
    quantiles_95 = pd.read_csv(shuffling.data_path('quantiles_95'))
    quantiles_95.head()
    action_columns = ['action', 'channel_group', 'unit_name']
    data = pd.merge(statistics, quantiles_95, on=action_columns, suffixes=("", "_threshold"))

    data['specificity'] = np.log10(data['in_field_mean_rate'] / data['out_field_mean_rate'])
    
    # waveform
    waveform_action = actions['waveform-analysis']
    waveform_results = pd.read_csv(waveform_action.data_path('results')).drop('template', axis=1)

    data = data.merge(waveform_results, how='left')

    data.bs = data.bs.astype(bool)

    data.loc[data.eval('t_i_peak == t_i_peak and not bs'), 'ns_inhibited'] = True
    data.ns_inhibited.fillna(False, inplace=True)

    data.loc[data.eval('t_i_peak != t_i_peak and not bs'), 'ns_not_inhibited'] = True
    data.ns_not_inhibited.fillna(False, inplace=True)
    
    # if a neuron is significantly inhibited once, we count it as a ns_inhibited
    
    data.loc[data.unit_id.isin(data.query('ns_inhibited').unit_id.values), 'ns_inhibited'] = True
    # we alsochange label from not inhibted to inhibited
    data.loc[data.eval('ns_inhibited'), 'ns_not_inhibited'] = False
    data.loc[data.eval('ns_inhibited'), 'bs'] = False
#     data.loc[data.unit_id.isin(data.query('ns_not_inhibited').unit_id.values), 'ns_not_inhibited'] = True
    
    # gridcells
    sessions_above_threshold = data.query(gridcell_query)
    print("Number of sessions above threshold", len(sessions_above_threshold))
    print("Number of animals", len(sessions_above_threshold.groupby(['entity'])))

    gridcell_sessions = data[data.unit_day.isin(sessions_above_threshold.unit_day.values)]
    print("Number of individual gridcells", gridcell_sessions.unit_idnum.nunique())
    print("Number of gridcell recordings", len(gridcell_sessions))
    data.loc[:,'gridcell'] = np.nan
    data['gridcell'] = data.isin(gridcell_sessions)

    data.loc[data.eval('not gridcell and bs'), 'bs_not_gridcell'] = True
    data.bs_not_gridcell.fillna(False, inplace=True)
    
    for i, query in enumerate(queries):
        data.loc[data.eval(query), 'label'] = labels[i]
        data.loc[data.eval(query), 'label_num'] = label_nums[i]
        data.loc[data.eval(query), 'query'] = query 
        data.loc[data.eval(query), 'color'] = colors[i]
        
    data['cell_type'] = np.nan
    for cell_type in cell_types:
        data.loc[data.eval(cell_type), 'cell_type'] = cell_type
    
    return data, labels, colors, queries


def drop_duplicates_least_null(df, key):
    return df.loc[df.notnull().sum(1).groupby(df[key]).idxmax()]


def _make_paired_table(data, queries, labels, key, cell_type, drop_duplicates):
    results = []
    for query, label in zip(queries, labels):
        values = data.query(query + f' and {cell_type}').loc[:,['entity', 'unit_idnum', 'channel_group', 'date', key]]
        results.append(values.rename({key: label}, axis=1))

    results = reduce(
            lambda  left, right: pd.merge(left, right, on=['entity', 'unit_idnum', 'channel_group', 'date'], how='outer'), results)
    
    if drop_duplicates:
        results = drop_duplicates_least_null(results, 'unit_idnum')
        
    return results


def _make_paired_tables(data, queries, labels, keys, cell_type, drop_duplicates):
    results = {}
    for key in keys:
        results[key] = _make_paired_table(data, queries, labels, key, cell_type, drop_duplicates)
        
                
    return results


def make_paired_tables(data, keys, drop_duplicates=True, queries=None, labels=None, cell_types=None):
    if queries is None:
        queries, labels, _, _ = get_queries()
    if cell_types is None:
        cell_types = [
            'gridcell',
            'ns_inhibited', 
            'ns_not_inhibited',
            'bs',
            'bs_not_gridcell'
        ]

    results = {}
    for cell_type in cell_types:
        results[cell_type] = _make_paired_tables(data, queries, labels, keys, cell_type, drop_duplicates)
        
    return results, labels


def LMM(df, case, control, key='val', use_unit_id=True, method=['powell', "lbfgs", 'bfgs', 'cg', 'basinhopping']):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    dd = pd.DataFrame()
    dd[key] = df[case]
    dd['label'] = 0
    dd['entity'] = df['entity']

    ddd = pd.DataFrame()
    ddd[key] = df[control]
    ddd['entity'] = df['entity']
    if use_unit_id:
        dd['unit_idnum'] = df['unit_idnum']
        ddd['unit_idnum'] = df['unit_idnum']
    ddd['label'] = 1
    dff = pd.concat([dd, ddd]).replace([np.inf, -np.inf], np.nan).dropna().reset_index()

    if dff.empty:
        return [np.nan] * 4 + ['empty']
    
    if use_unit_id:
        vc = {'unit_idnum': '0 + C(unit_idnum)'}
    else:
        vc = None
        
    mdf = smf.mixedlm(f"{key} ~ label", dff, groups="entity", missing='drop', vc_formula=vc, re_formula='label').fit(method=method)    
    low, high = mdf.conf_int(alpha=0.05).iloc[1,:].values
    pval = mdf.pvalues[1]
    marker = ''
    if np.isnan(pval):
        marker = '*'
        mdf = smf.mixedlm(f"{key} ~ label", dff, groups="entity", missing='drop', re_formula='label').fit(method=method)
        low, high = mdf.conf_int(alpha=0.05).iloc[1,:].values
        pval = mdf.pvalues[1]
        
    if np.isnan(pval):
        marker = '**'
        mdf = smf.mixedlm(f"{key} ~ label", dff, groups="entity", missing='drop').fit(method=method)
        low, high = mdf.conf_int(alpha=0.05).iloc[1,:].values
        pval = mdf.pvalues[1]
    
    return pval, low, high, mdf, marker


def rename(name):
    return name.replace("_field", "-field").replace("_", " ").capitalize()


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


def ttest_ind(df, keys):
    '''
    ttest individial
    '''
    d1 = df[keys[0]].dropna()
    d2 = df[keys[1]].dropna()
    statistic, pvalue = scipy.stats.ttest_ind(
        d1, d2, equal_var=False)

    return statistic, pvalue

def ttest_rel(df, keys):
    '''
    ttest pairwise
    '''
    dff = df.loc[:,[keys[0], keys[1]]].dropna()
    if dff.empty:
        return [np.nan] * 2
    statistic, pvalue = scipy.stats.ttest_rel(
        dff[keys[0]], 
        dff[keys[1]])

    return statistic, pvalue

    
def normality(df, key):
    statistic, pvalue = scipy.stats.normaltest(
        df[key].dropna())

    return statistic, pvalue


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


def make_statistics_table(
    results, labels, lmm_test=True, wilcoxon_test=False, ttest_ind_test=False, ttest_rel_test=False, 
    mannwhitney_test=False, show_cohen_d=False, use_weighted_stats=True, normality_test=False, block_permutation_test=False):
    stat_formatted = pd.DataFrame()
    stat_values = pd.DataFrame()
    for key, df in results.items():
        Key = rename(key)
        for label in labels:
            if use_weighted_stats:
                average, sem, n = compute_weighted_mean_sem(df, label)
            else:
                average, sem, n = df[label].mean(), df[label].sem(), df[label].count()
            stat_formatted.loc[label, Key] = np.nan if np.isnan(average) else "{:.3g} ± {:.3g} ({})".format(average, sem, n)
            stat_values.loc[label, key] = average
            
            if normality_test:
                if df[label].count() < 8:
                    stat_formatted.loc[f'Normality {label}', Key] = np.nan
                else:
                    statistic, pval = normality(df, label)
                    if np.isnan(statistic):
                        stat_formatted.loc[f'Normality {label}', Key] = np.nan
                    else:
                        stat_formatted.loc[f'Normality {label}', Key] = "{:.3g}, {:.3g}".format(statistic, pval)

        for i, c1 in enumerate(labels):
            for c2 in labels[i+1:]:
                if wilcoxon_test:
                    statistic, pval, n = wilcoxon(df, [c1, c2])
                    stat_formatted.loc[f'Wilcoxon {c1} - {c2}', Key] = np.nan if np.isnan(statistic) else "{:.3g}, {:.3g}, ({})".format(statistic, pval, n)
                    stat_values.loc[f'Wilcoxon {c1} - {c2}', key] = pval
                    dff = df.loc[:,[c1, c2]].dropna()
                    m1, s1, m2, s2 = dff[c1].mean(), dff[c1].sem(), dff[c2].mean(), dff[c2].sem()
                    stat_formatted.loc[f'Paired summary {c1} - {c2}', Key] = np.nan if np.isnan(statistic) else "{:.3g} ± {:.3g}, {:.3g} ± {:.3g}".format(m1, s1, m2, s2)
                
                if ttest_ind_test:
                    statistic, pval = ttest_ind(df, [c1, c2])
                    stat_formatted.loc[f'T test ind {c1} - {c2}', Key] = np.nan if np.isnan(statistic) else "{:.3g}, {:.3g}".format(statistic, pval)
                    stat_values.loc[f'T test ind {c1} - {c2}', key] = pval
                    
                if ttest_rel_test:
                    statistic, pval = ttest_rel(df, [c1, c2])
                    stat_formatted.loc[f'T test pair {c1} - {c2}', Key] = np.nan if np.isnan(statistic) else "{:.3g}, {:.3g}".format(statistic, pval)
                    stat_values.loc[f'T test pair {c1} - {c2}', key] = pval
                    
                if mannwhitney_test:
                    statistic, pval = MWU(df, [c1, c2])
                    stat_formatted.loc[f'MWU {c1} - {c2}', Key] = np.nan if np.isnan(statistic) else "{:.3g}, {:.3g}".format(statistic, pval)
                    stat_values.loc[f'MWU {c1} - {c2}', key] = pval
                    
                if lmm_test:
                    try:
                        pval, low, high, mdf, marker = LMM(df, c1, c2)
                    except np.linalg.LinAlgError:
                        pval = np.nan
                    try:
                        coef = mdf.fe_params[1]
                    except:
                        coef = np.nan
                    if marker=='empty':
                        lmm_res = marker
                    elif np.isnan(pval):
                        lmm_res = np.nan
                    else: 
                        lmm_res = r"\beta={:.3g}, p={:.3g}{}".format(coef, pval, marker)
                    stat_formatted.loc[f'LMM {c1} - {c2}', Key] = lmm_res
                    stat_values.loc[f'{c1} - {c2}', key] = pval
                    
                if show_cohen_d: # wrong for imbalanced data
                    cohen_d = (df[c1].mean() - df[c2].mean()) / np.sqrt(np.mean([df[c1].var(), df[c2].var()]))
                    stat_formatted.loc[f'Cohen`s d {c1} - {c2}', Key] = cohen_d
                    
                
    return stat_formatted, stat_values


class Resample:
    def __init__(self, df, groupby, labels):
        self.df = df
        self.groups = df.groupby(groupby)
        self.kdes = {}
        for entity, group in self.groups:
            self.kdes[entity] = {}
            for label in labels:
                vals = group[label].dropna().values
                if len(vals) < 2:
                    continue
                self.kdes[entity][label] = scipy.stats.gaussian_kde(vals)
                
    
    def resample(self, case, control, effect_size, total_n_samples=None):
        df = pd.DataFrame()
        kdes = self.kdes
        n_samples_case = np.array([kdes[entity][case].neff if case in kdes[entity] else 0 for entity in kdes], dtype=int)
        n_samples_control = np.array([kdes[entity][control].neff if control in kdes[entity] else 0 for entity in kdes], dtype=int)
        if total_n_samples is not None:
            n_samples_case = (n_samples_case / sum(n_samples_case) * total_n_samples).astype(int)
            n_samples_control = (n_samples_control / sum(n_samples_control) * total_n_samples).astype(int)
        for i, entity in enumerate(kdes):
            if case in kdes[entity]:
                case_values = kdes[entity][case].resample(n_samples_case[i]).ravel()
                case_values = case_values - case_values.mean() + effect_size
            else:
                case_values = []
            if control in kdes[entity]:
                control_values = kdes[entity][control].resample(n_samples_control[i]).ravel()
                control_values = control_values - control_values.mean()
            else:
                control_values = []
            entities = [int(entity)] * max(len(case_values), len(control_values))
            df_entity = pd.DataFrame([case_values, control_values, entities], index=[case,control,'entity']).T
            df = pd.concat([df, df_entity])
        return df


def estimate_power_lmm(df, case, control, effect_range=(0.1, 0.6, 0.1), n_samples=100, key='vals'):
    r = Resample(df, 'entity', [case, control])
    
    effect_sizes = np.arange(*effect_range)
    power = []
    for effect_size in tqdm(effect_sizes, desc=key.replace('_', ' ').capitalize()):
        ps_lmm = []
        for _ in range(n_samples):
            df = r.resample(control, case, effect_size)
            try:
                pvalue, _, _, _ = LMM(df, case, control, key, use_unit_id=False)
            except np.linalg.LinAlgError:
                pvalue = np.nan
            
            ps_lmm.append(pvalue)
        power.append(np.mean(np.array(ps_lmm) < 0.05))
    return power, effect_sizes


def _compute_p(r, control, case, n_repeats, effect_size, n_samples=None):
    ps_lmm = []
    for _ in range(n_repeats):
        df = r.resample(control, case, effect_size, n_samples)
        try:
            pvalue, _, _, _ = LMM(df, case, control, 'vals', use_unit_id=False)
        except np.linalg.LinAlgError:
            pvalue = np.nan
        ps_lmm.append(pvalue)
    return np.mean(np.array(ps_lmm) < 0.05), len(df)


def estimate_power_lmm_paralell(df, case, control, effect_range=(0.1, 0.6, 0.1), n_repeats=100):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
      
    effect_sizes = np.arange(*effect_range)
    r = Resample(df, 'entity', [case, control])
    
    power_eff_samples = pool.starmap(_compute_p, [(r, control, case, n_repeats, effect_size) for effect_size in effect_sizes])
    pool.close()
    
    power = [a[0] for a in power_eff_samples]
    effective_samples = [a[1] for a in power_eff_samples]
        
    return power, effect_sizes


def estimate_sample_size_lmm(df, case, control, effect_size, n_samples_range=(10, 200, 5), n_repeats=100, key='vals'):
    r = Resample(df, 'entity', [case, control])
    
    n_samples = np.arange(*n_samples_range).astype(int)
    power = []
    effective_samples = []
    for n_sample in tqdm(n_samples, desc=key.replace('_', ' ').capitalize()):
        ps_lmm = []
        for _ in range(n_repeats):
            df = r.resample(control, case, effect_size, n_sample)
            try:
                pvalue, _, _, _ = LMM(df, case, control, key, use_unit_id=False)
            except np.linalg.LinAlgError:
                pvalue = np.nan
                
            ps_lmm.append(pvalue)
        power.append(np.mean(np.array(ps_lmm) < 0.05))
        effective_samples.append(len(df))
    return power, n_samples, effective_samples


def estimate_sample_size_lmm_paralell(df, case, control, effect_size, n_samples_range=(10, 200, 5), n_repeats=100, key='vals'):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
      
    n_samples = np.arange(*n_samples_range).astype(int)
    r = Resample(df, 'entity', [case, control])
    
    power_eff_samples = pool.starmap(_compute_p, [(r, control, case, n_repeats, effect_size, n_sample) for n_sample in n_samples])    
    pool.close()
    
    power = [a[0] for a in power_eff_samples]
    effective_samples = [a[1] for a in power_eff_samples]
        
    return power, n_samples, effective_samples

######################################################################################################################################
# def bootstrap_results(results, labels, n_boots=100, n_samples=10, n_blocks=4):
#     bootstrapped_results = {}
#     for key, df in results.items():
#         bootstrapped_results[key] = pd.DataFrame()
#         group = df.groupby('entity')
#         for label in labels:
#             entity_values = np.array([d.loc[:, label].dropna().values for _, d in group if d.loc[:, label].count() > 0])
#             if len([i for j in entity_values for i in j]) < 3: # less than total 3 samples
#                     boot_samples = np.ones(n_boots) * np.nan
#             else:
#                 boot_samples = block_bootstrap(entity_values, n_boots=n_boots, n_samples=n_samples, n_blocks=n_blocks, statistic=np.mean)
#             bootstrapped_results[key].loc[:, label] = np.ravel(boot_samples)
#     return bootstrapped_results
                
    
# def compute_weighted_mean_sem(data, label, groupby='entity'):
#     group = data.groupby(groupby)
#     tmp = [d.loc[:, label].dropna().values for _, d in group]
#     values = np.concatenate(tmp)
#     if len(values) == 0:
#         return [np.nan] * 3
#     weights = np.concatenate([np.ones_like(a) / len(a) for a in tmp])
#     average = np.average(values, weights=weights)
#     # Fast and numerically precise:
#     variance = np.average((values - average)**2, weights=weights)
#     sem = np.sqrt(variance / len(values))
#     return average, sem, len(values)


# def compute_confidence_interval(data, alpha=0.05):
# #     stat = np.sort(data.dropna())
# #     n = len(stat)
# #     if n == 0:
# #         return np.nan, np.nan
# #     low = stat[int((alpha / 2.0) * n)]
# #     high = stat[int((1 - alpha / 2.0) * n)]
#     low, high = np.percentile(data.dropna(), [(alpha / 2.0) * 100, (1 - alpha / 2.0) * 100])
#     return low, high

# def make_bootstrap_table(results, bootstrapped_results, labels):
#     stat_formatted = pd.DataFrame()
#     stat_values = pd.DataFrame()
    
#     for key, df in bootstrapped_results.items():
#         Key = rename(key)

#         for label in labels:
#             average, sem, n = compute_weighted_mean_sem(results[key], label)
#             if np.isnan(average):
#                 stat_formatted.loc[label, Key] = np.nan
#                 stat_values.loc[label, key] = np.nan
#             else:
#                 stat_formatted.loc[label, Key] = "{:.1e} ± {:.1e} ({})".format(average, sem, n)
#                 stat_values.loc[label, Key] = average

#         for i, c1 in enumerate(df.columns):
#             for c2 in df.columns[i+1:]:
#                 pval, low, high = pvalue(results[key], df, c1, c2)
#                 if np.isnan(pval):
#                     stat_formatted.loc[f'{c1} - {c2}', Key] = np.nan
#                     stat_values.loc[f'{c1} - {c2}', key] = np.nan
#                 else:
#                     stat_formatted.loc[f'{c1} - {c2}', Key] = "{:.1e} [{:.1e}, {:.1e}]".format(pval, low, high)
#                     stat_values.loc[f'{c1} - {c2}', key] = pval
                
#     return stat_formatted, stat_values


# def pvalue(df, df_bootstrap, control_key, case_key):
#     '''
#     pvalue from bootstrap results, shifts the bootstrapped distribution
#     '''
#     case, b = df_bootstrap[case_key].dropna(), df_bootstrap[control_key].dropna()
#     if len(case) == 0 or len(b) == 0:
#         return [np.nan] * 3
    
#     n = len(case)
    
#     average_case, _, _ = compute_weighted_mean_sem(df, case_key)
#     average_control, _, _ = compute_weighted_mean_sem(df, control_key)
    
#     low, high = compute_confidence_interval(average_control - case)
    
#     case_shift = case - case.mean()
#     diff = abs(average_case - average_control)    
    
#     pval = (np.sum(case_shift > diff) + np.sum(case_shift < - diff)) / n
    
#     return pval, low, high


# def bootstrap_pvalue(df, control_key, case_key, n_boots=100, n_samples=10, n_blocks=4, delta=0.0, alpha=0.05):
#     '''
#     pvalue from bootstrap results, shifts the sample distribution
#     '''
#     average_case, _, _ = compute_weighted_mean_sem(df, case_key)
#     average_control, _, _ = compute_weighted_mean_sem(df, control_key)
#     group = df.groupby('entity')
#     case_values = np.array([d.loc[:, case_key].dropna().values for _, d in group if d.loc[:, case_key].count() > 0])
#     control_values = np.array([d.loc[:, control_key].dropna().values for _, d in group if d.loc[:, control_key].count() > 0])
    
#     case_boot = block_bootstrap(case_values - average_case + delta, n_boots=n_boots, n_samples=n_samples, n_blocks=n_blocks, statistic=lambda x:np.ravel(x))
#     control_boot = block_bootstrap(control_values - average_control, n_boots=n_boots, n_samples=n_samples, n_blocks=n_blocks, statistic=lambda x:np.ravel(x))
    
#     observed_diff = abs(average_case - average_control)
    
#     print(case_boot.shape)
    
#     # direct
#     diffs = case_boot.mean(1) - control_boot.mean(1)
#     low, high = np.percentile(diffs, [(alpha / 2.0) * 100, (1 - alpha / 2.0) * 100])
#     pval = (np.sum(diffs > observed_diff) + np.sum(diffs < - observed_diff)) / len(case_boot)
    
#     # tstat
# #     T = [scipy.stats.ttest_ind(a, b).statistic for a, b in zip(case_boot, control_boot)]
# #     pval_t = (1 + np.sum(np.abs(T) > np.abs(scipy.stats.ttest_ind(df.loc[:, case_key].dropna(), df.loc[:, control_key].dropna()).statistic))) / (len(case_boot)+1)

#     low_, high_ = np.percentile(average_control - case_boot.mean(1) + average_case, [(alpha / 2.0) * 100, (1 - alpha / 2.0) * 100])
    
#     pval_ = (np.sum(case_boot.mean(1) > observed_diff) + np.sum(case_boot.mean(1) < - observed_diff)) / len(case_boot)
    
#     return pval, low, high, pval_, low_, high_ 


# def power(df, control_key, case_key, delta, pval=0.05, n_boots=100, n_samples=10, n_blocks=4, ntest=100):
    
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