import numpy as np

def find_bursts(spike_times, threshold):
    if len(spike_times) <= 1:
        return [np.nan]*3
        
    isi = np.diff(spike_times)

    bursts = (isi[:-1] > threshold) & (isi[1:] <= threshold)
    bursts = np.append(isi[0] <= threshold, bursts)
    bursts = np.append(bursts, False)

    single_spikes = (isi[:-1] > threshold) & (isi[1:] > threshold)
    single_spikes = np.append(isi[0] > threshold, single_spikes)
    single_spikes = np.append(single_spikes, True)

    bursty = (isi[:-1] <= threshold) | (isi[1:] <= threshold)
    bursty = np.append(isi[0] <= threshold, bursty)
    bursty = np.append(bursty, isi[-1] <= threshold)

    return single_spikes, bursts, bursty
