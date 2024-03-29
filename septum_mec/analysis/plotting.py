import numpy as np
import matplotlib.pyplot as plt
from spike_statistics.core import bootstrap_ci, permutation_resampling_test
import scipy

    
def savefig(*args, **kwargs):
    plt.savefig(args[0].with_suffix('.png'), *args[1:], **kwargs, bbox_inches="tight", transparent=True, dpi=300)
    plt.savefig(args[0].with_suffix('.svg'), *args[1:], **kwargs, bbox_inches="tight", transparent=True)

def plot_psth(spike_times, stim_times, start_time, stop_time, binsize):
    fig, axs = plt.subplots(2, 1, sharex=True)
    trials = [spike_times[(spike_times > t + start_time) & (spike_times <= t + stop_time)] - t for t in stim_times]
    trials_ids = [[i]*len(t) for i, t in enumerate(trials)]

    trials_flat = [s for t in trials for s in t]
    trials_ids_flat = [s for t in trials_ids for s in t]

    axs[1].scatter(trials_flat, trials_ids_flat, s=1)
    bins = np.arange(start_time, stop_time + binsize, binsize)
    axs[0].hist(trials_flat, bins=bins);


def plot_waveforms(sptr, fig=None, gs=None, f=None, n=None, **kwargs):
    """
    Visualize waveforms on respective channels

    Parameters
    ----------
    sptr : neo.SpikeTrain
    color : color of waveforms
    title : figure title
    fig : matplotlib figure

    Returns
    -------
    out : fig
    """
    f = f if f is not None else lambda x, axis: x.T
    import matplotlib.gridspec as gridspec
    nrc = sptr.waveforms.shape[1]
    if fig is None:
        fig = plt.figure()
    axs = []
    ax = None
    for c in range(nrc):
        if gs is None:
            ax = fig.add_subplot(1, nrc, c+1, sharex=ax, sharey=ax)
        else:
            gs0 = gridspec.GridSpecFromSubplotSpec(1, nrc, subplot_spec=gs)
            ax = fig.add_subplot(gs0[:, c], sharex=ax, sharey=ax)
        axs.append(ax)
        wf = sptr.waveforms[:, c, :]
        n_spikes = wf.shape[0]
        stime = np.arange(wf.shape[1], dtype=np.float32) / sptr.sampling_rate
        stime.units = 'ms'
        if n is not None:
            idxs = np.random.choice(np.arange(n_spikes), size=n)
        else:
            idxs = np.arange(n_spikes)

        axs[c].plot(stime, f(wf[idxs,:], axis=0), **kwargs)
        m = np.mean(wf[idxs,:], axis=0)
        sd = np.std(wf[idxs,:], axis=0)
        axs[c].fill_between(stime, m-sd, m+sd, alpha=.1, color=kwargs.get('color'))
        axs[c].set_xlabel(stime.dimensionality)
        axs[c].set_xlim([stime.min(), stime.max()])
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
    axs[0].set_ylabel(r'amplitude $\pm$ std [%s]' % wf.dimensionality)
    return fig


def plot_bootstrap_timeseries(times, signals, num_samples=1000, statistic=None, alpha=0.05, ax=None, normalize_values=False, **kwargs):
    '''
    times : array
        time for signal
    signals : array or list of signals
        each row is a signal, each column is a timepoint
    num_samples : int
        The number of repetitions of random samples of your data.
    statistic : function(2darray, axis)
        The statistic you want to build the ci. Default is mean
    alpha : float
        confidence, 0.05 gives 95 %
    '''
    statistic = statistic or np.mean
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    cis = []
    values = np.array([statistic(signal) for signal in signals])
    if normalize_values:
        signals = signals - values.min()
        values = values - values.min()
        signals = signals / values.max()
        values = values / values.max()
    for signal in signals:
        ci = bootstrap_ci(signal, num_samples=num_samples, statistic=statistic)
        cis.append(ci)
    cis = np.array(cis)
    ax.plot(times, values, **kwargs)
    ax.fill_between(times, cis[:,0], cis[:,1], alpha=.5, color=kwargs.get('color'))


def plot_uncertainty(times, signals, ax=None, normalize_values=False, **kwargs):
    '''
    times : array
        time for signal
    signals : array or list of signals
        each row is a signal, each column is a timepoint
    num_samples : int
        The number of repetitions of random samples of your data.
    statistic : function(2darray, axis)
        The statistic you want to build the ci. Default is mean
    alpha : float
        confidence, 0.05 gives 95 %
    '''
    statistic = np.mean
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    cis = []
    values = np.array([statistic(signal) for signal in signals])
    if normalize_values:
        signals = signals - values.min()
        values = values - values.min()
        signals = signals / values.max()
        values = values / values.max()
    for signal in signals:
        sem = np.std(signal) / len(signal)
        cis.append([statistic(signal) - sem, statistic(signal) + sem])
    cis = np.array(cis)
    ax.plot(times, values, **kwargs)
    ax.fill_between(times, cis[:,0], cis[:,1], alpha=.5, color=kwargs.get('color'))


def violinplot(control, stimulated, xticks=["Baseline  ", "  Stimulated"], test='wilcoxon', colors=None, draw_significance=True):
    if draw_significance:
        if test == 'mann_whitney':
            Uvalue, pvalue = scipy.stats.mannwhitneyu(control, stimulated, alternative='two-sided')
            print("U-test: U value", Uvalue, 'p value', pvalue)
        elif test == 'permutation_resampling':
            pvalue, observed_diff, diffs = permutation_resampling_test(control, stimulated, statistic=np.median)
            print("P-test: diff", observed_diff, 'p value', pvalue)
        elif test == 'wilcoxon':
            from scipy.stats import wilcoxon
            Uvalue, pvalue = wilcoxon(control, stimulated)
        else:
            raise KeyError('Unable to recognize {}'.format(test))
            
    colors = colors if colors is not None else ['#2166ac', '#b2182b']
    pos = [0.0, 0.6]

    violins = plt.violinplot([control, stimulated], pos, showmedians=True, showextrema=False)
    
    for i, b in enumerate(violins['bodies']):
        b.set_color(colors[i])
        b.set_alpha (0.8)

    for category in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        if category in violins:
            violins[category].set_color(['k', 'k'])
            violins[category].set_linewidth(2.0)
    plt.xticks(pos, xticks, rotation=45)
            
    if draw_significance:
        # significance
        if pvalue < 0.0001:
            significance = "****"
        elif pvalue < 0.001:
            significance = "***"
        elif pvalue < 0.01:
            significance = "**"
        elif pvalue < 0.05:
            significance = "*"
        else:
            significance = "ns"

        x1, x2 = pos
        data_max = np.max([max(control), max(stimulated)])
        data_min = np.min([min(control), min(stimulated)])
        y = data_max * 1.05
        h = 0.025 * (data_max - data_min)
        plt.plot([x1, x1, x2, x2], [y - h, y, y, y - h], c='k')
        plt.text((x1 + x2) / 2, y + h, significance, ha='center', va='bottom')
        
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


def swarm_violin(x, data, ax=None, clip=None, color='k'):
    if ax is None:
        fig, ax = plt.subplots()
    sns.set_palette(palette=color)

    ticks = list(range(len(x)))
    data_list = [data[d].values for d in x]

    violins = ax.violinplot(
        data_list, ticks, showmedians=True, showextrema=False, points=1000, bw_method=.3)

    for category in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        if category in violins:
            violins[category].set_color(['w', 'w'])
            violins[category].set_linewidth(2.0)
            violins[category].set_zorder(10000)

    for c, pc in zip(color, violins['bodies']):
        pc.set_facecolor(c)
#         pc.set_edgecolor(c)
        pc.set_alpha(0.4)

    sns.stripplot(data=data_list, size=4, ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    y = -np.inf
    if clip is None:
        for val in data_list:
            data_max = max(val)
            data_min = min(val)
            y_ = data_max * 1.05 + 0.025 * (data_max - data_min)
            if y_ > y:
                y = y_
    else:
        y = clip
        ax.set_ylim(0, clip)

    x = 1
    for val in data_list[1:]:
        Uvalue, pvalue = wilcoxon(
            data_list[0] - val, alternative='less')
        # significance
        if pvalue < 0.0001:
            significance = "****"
        elif pvalue < 0.001:
            significance = "***"
        elif pvalue < 0.01:
            significance = "**"
        elif pvalue < 0.05:
            significance = "*"
        else:
            significance = "ns"

        ax.text(x, y, significance, ha='center', va='bottom')
        x += 1


def regplot(x, y, data=None, model=None, ci=95., scatter_color=None, model_color='k', ax=None,
            scatter_kws={}, regplot_kws={}, cmap=None, cax=None, clabel=None,
            xlabel=False, ylabel=False, colorbar=False, **kwargs):
    if model is None:
        import statsmodels.api as sm
        model = sm.OLS
    from seaborn import utils
    from seaborn import algorithms as algo
    if ax is None:
        fig, ax = plt.subplots()
    if data is None:
        _x = x
        _y = y
    else:
        _x = data[x]
        _y = data[y]
    grid = np.linspace(_x.min(), _x.max(), 100)

    X = np.c_[np.ones(len(_x)), _x]
    G = np.c_[np.ones(len(grid)), grid]

    results = model(_y, X, **kwargs).fit()

    def reg_func(xx, yy):
        yhat = model(yy, xx, **kwargs).fit().predict(G)
        return yhat
    yhat = results.predict(G)
    yhat_boots = algo.bootstrap(
        X, _y, func=reg_func, n_boot=1000, units=None)
    err_bands = utils.ci(yhat_boots, ci, axis=0)
    ax.plot(grid, yhat, color=model_color, **regplot_kws)
    sc = ax.scatter(_x, _y, c=scatter_color, **scatter_kws)
    ax.fill_between(grid, *err_bands, facecolor=model_color, alpha=.15)
    if colorbar:
        cb = plt.colorbar(mappable=sc, cax=cax, ax=ax)
        cb.ax.yaxis.set_ticks_position('right')
        if clabel: cb.set_label(clabel)

    if xlabel:
        if isinstance(xlabel, str):
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)
    if ylabel:
        if isinstance(ylabel, str):
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)
    return results


def despine(ax=None, left=False, right=True, top=True, bottom=False,
            xticks=True, yticks=True):
    """
    Removes axis lines
    """
    if ax is None:
        ax = plt.gcf().get_axes()
    if not isinstance(ax, (list, tuple)):
        ax = [ax]
    for a in ax:
        try:
            a.spines['top'].set_visible(not top)
            a.spines['right'].set_visible(not right)
            a.spines['left'].set_visible(not left)
            a.spines['bottom'].set_visible(not bottom)
        except KeyError:
            pass

        a.get_xaxis().tick_bottom()
        plt.setp(a.get_xticklabels(), visible=xticks)
        if not xticks:
            a.xaxis.set_ticks_position('none')
        a.get_yaxis().tick_left()
        plt.setp(a.get_yticklabels(), visible=yticks)
        if not yticks:
            a.yaxis.set_ticks_position('none')


def lighten_color(color, amount=0.7):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


from seaborn.categorical import _ViolinPlotter
from seaborn.utils import remove_na


class MyVPlot(_ViolinPlotter):

    def draw_to_density(self, ax, center, val, support, density, split, **kws):
        """Draw a line orthogonal to the value axis at width of density."""
        idx = np.argmin(np.abs(support - val))
        width = self.dwidth * density[idx] * .99

        kws["color"] = self.gray

        offset = width * 0.1 # draw within edges

        if self.orient == "v":
            if split == "left":
                ax.plot([center - width + offset, center - offset], [val, val], **kws)
            elif split == "right":
                ax.plot([center + offset, center + width - offset], [val, val], **kws)
            else:
                ax.plot([center - width, center + width], [val, val], **kws)
        else:
            if split == "left":
                ax.plot([val, val], [center - width + offset, center - offset], **kws)
            elif split == "right":
                ax.plot([val, val], [center + offset, center + width - offset], **kws)
            else:
                ax.plot([val, val], [center - width, center + width], **kws)

    def draw_quartiles(self, ax, data, support, density, center, split=False):
        """Draw the quartiles as lines at width of density."""
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        self.draw_to_density(ax, center, q50, support, density, split,
                             linewidth=2, )

    def draw_significance(self, ax, test='mann_whitney'):
        significance = {'top': -np.inf, 'val': {}}
        for i, group_data in enumerate(self.plot_data):
            tmp_data = []
            for j, hue_level in enumerate(self.hue_names):
                hue_mask = self.plot_hues[i] == hue_level
                violin_data = remove_na(group_data[hue_mask])
                tmp_data.append(violin_data)

            if test == 'mann_whitney':
                Uvalue, pvalue = scipy.stats.mannwhitneyu(*tmp_data, alternative='two-sided')
            elif test == 'permutation_resampling':
                pvalue, observed_diff, diffs = permutation_resampling_test(*tmp_data, statistic=np.median)
            else:
                raise KeyError('Unable to recognize {}'.format(test))

            # significance
            if pvalue < 0.0001:
                symbol = "****"
            elif pvalue < 0.001:
                symbol = "***"
            elif pvalue < 0.01:
                symbol = "**"
            elif pvalue < 0.05:
                symbol = "*"
            else:
                symbol = "ns"

            significance['val'][i] = symbol

            data_max = np.max([max(tmp_data[0]), max(tmp_data[1])]) * 1.05
#             data_min = np.min([min(tmp_data[0]), min(tmp_data[1])])
#             y = data_max * 1.05
#             h = 0.025 * (data_max - data_min)

            if data_max > significance['top']:
                significance['top'] = data_max

        for i, s in significance['val'].items():
            plt.text(i, significance['top'], s, ha='center', va='bottom')


def split_violinplot(x=None, y=None, hue=None, data=None, order=None, hue_order=None,
                  bw="scott", cut=0, scale="area", scale_hue=True, gridsize=100,
                  width=.8, inner="box", split=False, dodge=True, orient=None,
                  linewidth=None, color=None, palette=None, saturation=.75,
                  ax=None, draw_significance=True, **kwargs):
    plotter = MyVPlot(x, y, hue, data, order, hue_order,
                      bw, cut, scale, scale_hue, gridsize,
                      width, inner, split, dodge, orient, linewidth,
                      color, palette, saturation)
    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)
    if draw_significance:
        plotter.draw_significance(ax)

    for i, c in enumerate(ax.collections):
        fc = c.get_facecolor()
        c.set_edgecolor(fc)
        c.set_alpha(0.8)

    return ax
