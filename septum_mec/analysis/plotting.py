import numpy as np
import matplotlib.pyplot as plt
from spike_statistics.core import bootstrap, permutation_resampling


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


def plot_bootstrap_timeseries(times, signals, num_samples=1000, statistic=None, alpha=0.05, ax=None, **kwargs):
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
    values = []
    for signal in signals:
        ci = bootstrap(signal, num_samples=num_samples, statistic=statistic)
        cis.append(ci)
        values.append(statistic(signal))
    cis = np.array(cis)
    ax.plot(times, values, **kwargs)
    ax.fill_between(times, cis[:,0], cis[:,1], alpha=.5, color=kwargs.get('color'))


def violinplot(control, chabc, xticks=["Control", "chABC"], test='mann_whitney'):
    if test == 'mann_whitney':
        Uvalue, pvalue = scipy.stats.mannwhitneyu(control, chabc, alternative='two-sided')
        print("U-test: U value", Uvalue, 'p value', pvalue)
    elif test == 'permutation_resampling':
        pvalue, observed_diff, diffs = permutation_resampling(control, chabc, statistic=np.median)
        print("P-test: diff", observed_diff, 'p value', pvalue)
    else:
        raise KeyError('Unable to recognize {}'.format(test))

    pos = [0.0, 0.6]

    violins = plt.violinplot([control, chabc], pos, showmedians=True, showextrema=False)

#     for i, body in enumerate(violins['bodies']):
#         body.set_color('C{}'.format(i))
#         body.set_linewidth(2)#         body.set_linewidth(2)
    violins['bodies'][0].set_color ('#2166ac')
    violins['bodies'][1].set_color ('#b2182b')
    violins['bodies'][0].set_color ('#053061')
    violins['bodies'][1].set_color ('#67001f')
    violins['bodies'][0].set_color ('#4393c3')
    violins['bodies'][1].set_color ('#d6604d')
    violins['bodies'][0].set_alpha (0.8)
    violins['bodies'][1].set_alpha (0.8)

    # for i, body in enumerate(violins['cbars']):
    #     body.set_color('C{}'.format(i))

    for category in ['cbars', 'cmins', 'cmaxes', 'cmedians']:
        if category in violins:
            violins[category].set_color(['k', 'k'])
            violins[category].set_linewidth(2.0)

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

    plt.xticks(pos, xticks)

    x1, x2 = pos
    data_max = np.max([max(control), max(chabc)])
    data_min = np.min([min(control), min(chabc)])
    y = data_max * 1.05
    h = 0.025 * (data_max - data_min)
    plt.plot([x1, x1, x2, x2], [y - h, y, y, y - h], c='k')
    plt.text((x1 + x2) / 2, y + h, significance, ha='center', va='bottom')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)


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

        if not xticks:
            a.get_xaxis().tick_bottom()
            plt.setp(a.get_xticklabels(), visible=False)
        if not yticks:
            a.get_yaxis().tick_left()
            plt.setp(a.get_yticklabels(), visible=False)
