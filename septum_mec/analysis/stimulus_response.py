import numpy as np
from scipy.signal import find_peaks

def spike_stim_count(spike_times, stim_times, time, limit):
    stim_win = np.insert(
        stim_times + time - limit,
        np.arange(len(stim_times)) + 1,
        stim_times + time + limit)
    src_y = np.searchsorted(spike_times, stim_win, side='left')
    cnt_y = np.diff(src_y.reshape((int(len(src_y) / 2), 2)))
    return cnt_y.flatten()

def compute_response(spike_times, stim_times, times, kernel, e_percentile, i_percentile, limit=1e-3):
    hist = kernel(times)
    p_e = hist > e_percentile
    p_i = hist < i_percentile
    idxs_e, _ = find_peaks(hist)
    # significant peaks
    idxs_e = idxs_e[p_e[idxs_e]]
    te_peak, pe_peak = np.nan, np.nan
    if len(idxs_e) > 0:
        # pick the largest
        idxs_e = idxs_e[np.argmax(hist[idxs_e])]
        te_peak = times[idxs_e]
        pe_cnt = spike_stim_count(spike_times, stim_times, te_peak, limit)
        pe_peak = sum(pe_cnt > 0) / len(stim_times)

    idxs_i, _ = find_peaks(- hist)
    # significant peaks
    idxs_i = idxs_i[p_i[idxs_i]]
    # only allow inhibition before excitation
    ti_peak, pi_peak = np.nan, np.nan
    if any(times[idxs_i] < te_peak):
        idxs_i = idxs_i[times[idxs_i] < te_peak]
        # pick the smallest
        idxs_i = idxs_i[np.argmin(hist[idxs_i])]
        ti_peak = times[idxs_i]
        pi_cnt = spike_stim_count(spike_times, stim_times, ti_peak, limit)
        pi_peak = sum(pi_cnt > 0) / len(stim_times)
    return te_peak, pe_peak, ti_peak, pi_peak


def stimulus_response_latency(spike_times, stim_times, window, std, percentile=99, plot=False):
    from scipy.stats import gaussian_kde
    spike_times = np.array(spike_times)
    stim_times = np.array(stim_times)
    n_spikes, n_stim = len(spike_times), len(stim_times)
    times = np.arange(0, window, 1e-4)
    trials = [spike_times[(spike_times >= t - window) & (spike_times <= t + window)] - t
              for t in stim_times]
    spikes = [s for t in trials for s in t]
    if len(spikes) == 0:
        return [np.nan] * 5
    kernel = gaussian_kde(spikes, std)

    # we start 10 % away from -window due to edge effects
    pre_times = np.arange(- window + window * 0.1, 0, 1e-4)
    i_percentile = np.percentile(kernel(pre_times), 100 - percentile, 0)
    e_percentile = np.percentile(kernel(pre_times), percentile, 0)
    if plot:
        import matplotlib.pyplot as plt
        all_times = np.arange(-window, window, 1e-4)
        plt.plot(all_times, kernel(all_times))
        plt.plot(pre_times, kernel(pre_times))
        plt.plot(times, [i_percentile] * len(times))
        plt.plot(times, [e_percentile] * len(times))


    return times, spikes, kernel, e_percentile, i_percentile
