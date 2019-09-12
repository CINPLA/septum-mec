from matplotlib.mlab import psd
import matplotlib.pyplot as plt
import pycwt
from septum_mec.analysis.plotting import plot_bootstrap_timeseries, despine
import spatial_maps as sp
import head_direction as hd
import septum_mec.analysis.data_processing as dp
import expipe
import os
import pathlib
import math
import numpy as np
import exdir
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib import gridspec
import quantities as pq
import elephant as el
import speed_cells.speed as spd
import seaborn as sns
import neo


def spatial(x, y, t, sptr, a, at, binsize, smoothing, fig, gs, t_start=None, t_stop=None):
    mask_zero_occupancy = False

    if t_start is not None and t_stop is not None:
        mask = (t < t_stop) & (t > t_start)
        x, y, t = x[mask], y[mask], t[mask]
        sptr = neo.SpikeTrain(
            sptr.times[(sptr.times > t_start) & (sptr.times < t_stop)],
            t_start=t_start, t_stop=t_stop, **sptr.annotations
        )

    gs0 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs, wspace=.02, hspace=.02)

    spatial_map = sp.SpatialMap(
        x, y, t, sptr, box_size=1.0, bin_size=binsize)

    rate_map = spatial_map.rate_map(smoothing=smoothing, mask_zero_occupancy=mask_zero_occupancy)

    axs = [fig.add_subplot(g, projection='polar' if i == 2 else None) for i, g in enumerate(gs0)]
    axs[0].imshow(rate_map, vmin=0)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title('gridness {:.2f}'.format(sp.gridcells.gridness(rate_map)))

    axs[1].plot(x, y, 'k', alpha=0.3)
    # Remove spikes that fall outside the range of times.
    # Otherwise, interpolation will fail when plotting spikes in scatter plot.
    sptr = sptr[(sptr > min(t)) & (sptr < max(t))]
    x_spike = interp1d(t,x)(sptr)
    y_spike = interp1d(t,y)(sptr)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[1].scatter(x_spike, y_spike, marker='.', color=(0.7, 0.2, 0.2), s=2)
    axs[1].set_ylim(1,0)
    axs[1].set_title('N spikes {}'.format(len(sptr)))

#     np.nan_to_num(rate_map, copy = False) # inplace
#     autocorr = sp.tools.fftcorrelate2d(rate_map, rate_map, mode = 'full', normalize = True)
#     axs[2].imshow(autocorr, vmin=0)
#     axs[2].set_xticks([])
#     axs[2].set_yticks([])

    ang_bins, rate_ang = hd.head_direction_rate(sptr, a, at)
    hd_dir, hd_score = hd.head_direction_score(ang_bins, rate_ang)
    hd_dir = math.degrees(hd_dir)
    binsize = ang_bins[1] - ang_bins[0]
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[2].bar(ang_bins, rate_ang, width=binsize, color='b')
    axs[2].set_title('hd {:.2f}, {:.1f} deg'.format(hd_score, hd_dir))
    # plt.setp(axs[2].get_xticklabels(), visible=False)

    for ax in axs:
        ax.set_aspect(1)
    return axs


def speedcor(speed, t, spike_times, min_speed, max_speed, binsize, fig, gs, t_start=None, t_stop=None):

    if t_start is not None and t_stop is not None:
        mask = (t < t_stop) & (t > t_start)
        x, y, t = x[mask], y[mask], t[mask]
        spike_times = neo.SpikeTrain(
            spike_times.times[(spike_times.times > t_start) & (spike_times.times < t_stop)],
            t_start=t_start, t_stop=t_stop, **spike_times.annotations
        )

    gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs)
    ax = fig.add_subplot(gs0[0])

    speed_score, inst_speed, rate, times = spd.speed_correlation(
        speed, t, spike_times, return_data=True)

    speed_bins = np.arange(min_speed, max_speed + binsize, binsize)
    ia = np.digitize(inst_speed, bins=speed_bins, right=True)
    rates = []

    for i in range(len(speed_bins)):
        rates.append(rate[ia==i])

    ax.set_title('speed score {:.3f}'.format(speed_score))
    plot_bootstrap_timeseries(speed_bins, rates, ax=ax)
    ax.set_xlabel('Speed m/s')
    ax.set_ylabel('Rate spikes/s')

    # mean_rate = [np.mean(r) for r in rates]
    # aspect = (max_speed - min_speed) / (np.nanmax(mean_rate) - np.nanmin(mean_rate))
    # ax.set_aspect(aspect)
    return ax


def spikelfp(sptr, anas, fig, gs, t_start=None, t_stop=None, f_start=None, f_stop=None, axs_share=None):

    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs)
    if axs_share is not None:
        axs = [fig.add_subplot(g, sharey=ax) for g, ax in zip(gs0, axs_share)]
    else:
        axs = [fig.add_subplot(g) for g in gs0]

    sampling_rate = anas.sampling_rate.magnitude

    t_start = t_start * pq.s if t_start is not None else 0 * pq.s
    sampling_rate = anas.sampling_rate
    anot = anas.annotations
    units = anas.units
    if t_start is not None and t_stop is not None:
        t_stop = t_stop * pq.s
        mask = (anas.times > t_start) & (anas.times < t_stop)
        anas = np.array(anas)[mask,:]

    anas, _ = select_and_clean(anas)
    anas = neo.AnalogSignal(
        signal=anas * units, sampling_rate=sampling_rate, t_start=t_start,
        **anot
    )

    sptr = neo.SpikeTrain(
        sptr.times[(sptr.times > t_start) & (sptr.times < anas.times[-1])],
        t_start=t_start, t_stop=anas.times[-1],
        **sptr.annotations
    )

    sigs, freqs = el.sta.spike_field_coherence(anas, sptr, **{'nperseg': 2048})

    for sig, ch in zip(sigs.T, anot['electrode_idx']):
        axs[0].plot(freqs, sig, label='ch {}'.format(ch))
    # axs[0].legend(frameon=False, ncol=2)
    if f_start is not None and f_stop is not None:
        axs[0].set_xlim(f_start, f_stop)
    axs[0].set_ylabel('lfp coherence')
    despine(axs[0], xticks=False)

    sampling_rate = sampling_rate.magnitude
    bins = np.arange(
        sptr.t_start.magnitude, sptr.t_stop.magnitude, 1 / sampling_rate) #time bins for spikes
    #firing rate histogram
    hist = np.histogram(sptr, bins=bins)[0].astype(float)
    hist -= hist.mean()
    axs[1].psd(hist, Fs=sampling_rate, NFFT=2048)
    if f_start is not None and f_stop is not None:
        axs[1].set_xlim(f_start, f_stop)
    axs[1].set_ylabel('spike psd')
    return axs


def psth(sptr, stim_times, fig, gs, pre_stim, post_stim, binsize, t_start=None, t_stop=None):

    if t_start is not None and t_stop is not None:
        stim_times = stim_times[(stim_times > t_start) & (stim_times < t_stop)]
        sptr = neo.SpikeTrain(
            sptr.times[(sptr.times > t_start) & (sptr.times < t_stop)],
            t_start=t_start, t_stop=t_stop, **sptr.annotations
        )

    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs)
    axs = [fig.add_subplot(g) for g in gs0]

    spike_times = sptr.magnitude
    trials = [
        spike_times[(spike_times > t + pre_stim) & (spike_times <= t + post_stim)] - t
        for t in stim_times
    ]
    trials_ids = [[i]*len(t) for i, t in enumerate(trials)]

    trials_flat = [s for t in trials for s in t]
    trials_ids_flat = [s for t in trials_ids for s in t]
    axs[1].scatter(trials_flat, trials_ids_flat, s=.5)
    axs[1].set_xlim(pre_stim, post_stim)
    axs[1].set_xlabel('relative stim time')

    bins = np.arange(pre_stim, post_stim + binsize, binsize)
    hist, bins = np.histogram(trials_flat, bins=bins)
    hist = (hist / binsize) / len(stim_times)
    axs[0].bar(bins[:-1], hist, width=-binsize, align='edge');
    axs[0].set_xlim(pre_stim, post_stim)
    despine(axs[0], xticks=False)
    axs[0].set_ylabel('rate pr trial')
    return axs


def waveform(sptr, fig, gs, t_start=None, t_stop=None, axs=None, **kwargs):

    if t_start is not None and t_stop is not None:
        mask = (sptr.times > t_start) & (sptr.times < t_stop)
        sptr = neo.SpikeTrain(
            sptr.times[mask], waveforms=sptr.waveforms[mask, c, :],
            t_start=t_start, t_stop=t_stop, **sptr.annotations
        )

    nrc = sptr.waveforms.shape[1]
    if axs is None:
        gs0 = gridspec.GridSpecFromSubplotSpec(1, nrc, subplot_spec=gs)
        axs = [fig.add_subplot(g) for g in gs0]

    for c in range(nrc):
        wf = sptr.waveforms[:, c, :]
        stime = np.arange(wf.shape[1], dtype=np.float32) / sptr.sampling_rate
        stime = stime.rescale('ms').magnitude
        m = np.median(wf, axis=0)
        sd = np.std(wf, axis=0)
        axs[c].plot(stime, m, **kwargs)
        axs[c].fill_between(stime, m-sd, m+sd, alpha=.1, color=kwargs.get('color'))
#         plot_bootstrap_timeseries(stime, wf.T, ax=axs[c], statistic=np.median)
        axs[c].set_xlabel('ms')
        axs[c].set_xlim([stime.min(), stime.max()])
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
        else:
            axs[c].set_ylabel('amplitude uV')
    return axs


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def mask_coi(power, freqs, coi):
    periods = 1 / freqs
    for i in range(power.shape[1]):
        mask = periods > coi[i]
        power[mask, i] = np.nan


def wavelet(data, sampling_rate, f_start, f_stop, f_step=1, morlet=20):
    mother = pycwt.Morlet(morlet) # Morlet size
    freqs = np.arange(f_start, f_stop + f_step, f_step) # Frequency range

    wave, scales, freqs, coi, fft, fftfreqs = pycwt.cwt(
        data, 1. / sampling_rate, freqs=freqs, wavelet=mother)

    power = (np.abs(wave)) ** 2
    power /= scales[:, None] #rectify the power spectrum according to suggestions proposed by Liu et al. (2007)

    mask_coi(power, freqs, coi)

    return freqs, power


def select_and_clean(anas, width=500, threshold=2):
    anas = np.array(anas)

    channel = np.argmax(signaltonoise(anas, axis=0))
    data = anas[:, channel]
    idxs, = np.where(abs(data) > threshold)
    for idx in idxs:
        anas[idx-width:idx+width,:] = 0 # TODO AR model prediction
    return anas, channel


def tfr(anas, epochs, fig, gs, t_start=None, t_stop=None, f_start=None, f_stop=None, threshold=2):
    gs0 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs, width_ratios=[.7, 1], height_ratios=[1, .7],
        wspace=.02, hspace=.02)
    ax_psd = fig.add_subplot(gs0[0,0])
    ax_t = fig.add_subplot(gs0[1,1])
    ax_tfr = fig.add_subplot(gs0[0,1], sharey=ax_psd, sharex=ax_t)

    sampling_rate = anas.sampling_rate.magnitude

    times = np.arange(0, anas.shape[0] / sampling_rate, 1 / sampling_rate)

    anas, channel = select_and_clean(anas, threshold=threshold)
    data = anas[:, channel]
    if t_start is not None and t_stop is not None:
        mask = (t_start < times) & (times < t_stop) # seconds
        times, data = times[mask], data[mask]


    pxx, freqs = psd(data, Fs=sampling_rate, NFFT=4000)
    ax_psd.plot(pxx, freqs)
    ax_psd.set_ylim(f_start, f_stop)
    ax_psd.set_ylabel('Frequency Hz')

    ax_t.plot(times, data)

    if len(epochs) == 1:
        epo = epochs[0].times.magnitude
        ax_tfr.set_title('stimulation frequency {:.2f}'.format(1 / np.diff(epo).mean()))
        ax_t.axvspan(epo[0], epo[-1], alpha=.3)
    ax_t.set_xlim(t_start, t_stop)
    ax_t.set_xlabel('Time s')

    freqs, power = wavelet(data, sampling_rate, f_start, f_stop)
    ax_tfr.contourf(times, freqs, power, levels=100)

    despine(ax_tfr, xticks=False, yticks=False)
