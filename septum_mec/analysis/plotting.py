import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import neo


def plot_raster(trials, color="#3498db", lw=1, ax=None, marker='.', marker_size=10,
                ylabel='Trials', id_start=0, ylim=None):
    """
    Raster plot of trials
    Parameters
    ----------
    trials : list of neo.SpikeTrains
    color : color of spikes
    lw : line width
    ax : matplotlib axes
    Returns
    -------
    out : axes
    """
    from matplotlib.ticker import MaxNLocator
    if ax is None:
        fig, ax = plt.subplots()
    trial_id = []
    spikes = []
    dim = trials[0].times.dimensionality
    for n, trial in enumerate(trials):  # TODO what about empty trials?
        n += id_start
        spikes.extend(trial.times.magnitude)
        trial_id.extend([n]*len(trial.times))
    if marker_size is None:
        heights = 6000./len(trials)
        if heights < 0.9:
            heights = 1.  # min size
    else:
        heights = marker_size
    ax.scatter(spikes, trial_id, marker=marker, s=heights, lw=lw, color=color,
               edgecolors='face')
    if ylim is None:
        ax.set_ylim(-0.5, len(trials)-0.5)
    elif ylim is True:
        ax.set_ylim(ylim)
    else:
        pass
    y_ax = ax.axes.get_yaxis()  # Get X axis
    y_ax.set_major_locator(MaxNLocator(integer=True))
    t_start = trials[0].t_start.rescale(dim)
    t_stop = trials[0].t_stop.rescale(dim)
    ax.set_xlim([t_start, t_stop])
    ax.set_xlabel("Times [{}]".format(dim))
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax


def plot_psth(trials,
              fig=None, axs=None, legend_loc=1, color='b',
              stim_alpha=.2, stim_color=None,
              stim_label='Stim on', stim_duraton=None, stim_offset=0*pq.s,
              rast_ylabel='Trials', rast_size=10,
              hist_color=None, hist_edgecolor=None,
              hist_ylim=None,  hist_ylabel=None,
              hist_output='counts', hist_binsize=None, hist_nbins=100,
              hist_alpha=1.):
    """
    Visualize clustering on amplitude at detection point
    Parameters
    ----------
    trials : list of cut neo.SpikeTrains with same number of recording channels
    fig : matplotlib figure
    axs : matplotlib axes (must be 2)
    legend_loc : 'outside' or matplotlib standard loc
    color : color of spikes
    title : figure title
    stim_alpha : float
    stim_color : str
    stim_label : str
    stim_duraton : float
    stim_offset : pq.Quantity
        The amount of offset for the stimulus relative to epoch.
    rast_ylabel : str
    hist_color : str
    hist_edgecolor : str
    hist_ylim : list
    hist_ylabel : str
    hist_output : str
        Accepts 'counts', 'rate' or 'mean'.
    hist_binsize : pq.Quantity
    hist_nbins : int
    Returns
    -------
    out : fig
    """
    if fig is None and axs is None:
        fig, (hist_ax, rast_ax) = plt.subplots(2, 1, sharex=True)
    elif fig is not None and axs is None:
        hist_ax = fig.add_subplot(2, 1, 1)
        rast_ax = fig.add_subplot(2, 1, 2, sharex=hist_ax)
    else:
        assert len(axs) == 2
        hist_ax, rast_ax = axs

    dim = trials[0].times.dimensionality

    # raster
    plot_raster(trials, color=color, ax=rast_ax, ylabel=rast_ylabel,
                marker_size=rast_size)
    # histogram
    hist_color = color if hist_color is None else hist_color
    hist_ylabel = hist_output if hist_ylabel is None else hist_ylabel
    plot_spike_histogram(trials, color=hist_color, ax=hist_ax,
                         output=hist_output, binsize=hist_binsize,
                         nbins=hist_nbins, edgecolor=hist_edgecolor,
                         ylabel=hist_ylabel, alpha=hist_alpha)
    if hist_ylim is not None: hist_ax.set_ylim(hist_ylim)
    # stim representation
    stim_color = color if stim_color is None else stim_color
    if stim_duraton:
        fill_stop = stim_duration
        import matplotlib.patches as mpatches
        line = mpatches.Patch([], [], color=stim_color, label=stim_label,
                              alpha=stim_alpha)

        stim_offset = stim_offset.rescale(dim).magnitude
        hist_ax.axvspan(stim_offset, fill_stop + stim_offset, color=stim_color,
                        alpha=stim_alpha, zorder=0)
        rast_ax.axvspan(stim_offset, fill_stop + stim_offset, color=stim_color,
                        alpha=stim_alpha, zorder=0)
        if legend_loc == 'outside':
            hist_ax.legend(handles=[line], bbox_to_anchor=(0., 1.02, 1., .102),
                           loc=4, ncol=2, borderaxespad=0.)
        else:
            hist_ax.legend(handles=[line], loc=legend_loc, ncol=2, borderaxespad=0.)
    return fig


def plot_spike_histogram(trials, color='b', ax=None, binsize=None, bins=None,
                         output='counts', edgecolor=None, alpha=1., ylabel=None,
                         nbins=None):
    """
    histogram plot of trials

    Parameters
    ----------
    trials : list of neo.SpikeTrains
    color : str
        Color of histogram.
    edgecolor : str
        Color of histogram edges.
    ax : matplotlib axes
    output : str
        Accepts 'counts', 'rate' or 'mean'.
    binsize :
        Binsize of spike rate histogram, default None, if not None then
        bins are overridden.
    nbins : int
        Number of bins, defaults to 100 if binsize is None.
    ylabel : str
        The ylabel of the plot, if None defaults to output type.

    Examples
    --------
    >>> import neo
    >>> from numpy.random import rand
    >>> from exana.stimulus import make_spiketrain_trials
    >>> spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
    >>> epoch = neo.Epoch(times=np.arange(0, 10, 1) * pq.s,
    ...                   durations=[.5] * 10 * pq.s)
    >>> trials = make_spiketrain_trials(spike_train, epoch)
    >>> ax = plot_spike_histogram(trials, color='r', edgecolor='b',
    ...                           binsize=1 * pq.ms, output='rate', alpha=.5)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import quantities as pq
        import neo
        from numpy.random import rand
        from exana.stimulus import make_spiketrain_trials
        from statistics_plot import plot_spike_histogram
        spike_train = neo.SpikeTrain(rand(1000) * 10, t_stop=10, units='s')
        epoch = neo.Epoch(times=np.arange(0, 10, 1) * pq.s, durations=[.5] * 10 * pq.s)
        trials = make_spiketrain_trials(spike_train, epoch)
        ax = plot_spike_histogram(trials, color='r', edgecolor='b', binsize=1 * pq.ms, output='rate', alpha=.5)
        plt.show()

    Returns
    -------
    out : axes
    """
    ### TODO
    if bins is not None:
        assert isinstance(bins, int)
        warnings.warn('The variable "bins" is deprecated, use nbins in stead.')
        nbins = bins
    ###
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    from elephant.statistics import time_histogram
    dim = trials[0].times.dimensionality
    t_start = trials[0].t_start.rescale(dim)
    t_stop = trials[0].t_stop.rescale(dim)
    if binsize is None:
        if nbins is None:
            nbins = 100
        binsize = (abs(t_start)+abs(t_stop))/float(nbins)
    else:
        binsize = binsize.rescale(dim)
    time_hist = time_histogram(trials, binsize, t_start=t_start,
                               t_stop=t_stop, output=output, binary=False)
    bs = np.arange(t_start.magnitude, t_stop.magnitude, binsize.magnitude)
    if ylabel is None:
        if output == 'counts':
            ax.set_ylabel('count')
        elif output == 'rate':
            time_hist = time_hist.rescale('Hz')
            if ylabel:
                ax.set_ylabel('rate [%s]' % time_hist.dimensionality)
        elif output == 'mean':
            ax.set_ylabel('mean count')
    elif isinstance(ylabel, str):
        ax.set_ylabel(ylabel)
    else:
        raise TypeError('ylabel must be str not "' + str(type(ylabel)) + '"')
    ax.bar(bs[:len(time_hist)], time_hist.magnitude.flatten(), width=bs[1]-bs[0],
           edgecolor=edgecolor, facecolor=color, alpha=alpha, align='edge')
    return ax


def plot_waveforms(sptr, color='r', fig=None, title='waveforms', lw=2, gs=None):
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
    for c in range(nrc):
        wf = sptr.waveforms[:, c, :]
        m = np.mean(wf, axis=0)
        stime = np.arange(m.size, dtype=np.float32)/sptr.sampling_rate
        stime.units = 'ms'
        sd = np.std(wf, axis=0)
        axs[c].plot(stime, m, color=color, lw=lw)
        axs[c].fill_between(stime, m-sd, m+sd, alpha=.1, color=color)
        if sptr.left_sweep is not None:
            sptr.left_sweep.units = 'ms'
            axs[c].axvspan(sptr.left_sweep, sptr.left_sweep, color='k',
                           ls='--')
        axs[c].set_xlabel(stime.dimensionality)
        axs[c].set_xlim([stime.min(), stime.max()])
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
    axs[0].set_ylabel(r'amplitude $\pm$ std [%s]' % wf.dimensionality)
    fig.suptitle(title)
    return fig
