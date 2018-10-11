import os
import os.path as op
import numpy as np
from datetime import datetime
import quantities as pq
from septum_mec.tools.utils import read_python


def apply_CAR(anas, channels=None, car_type='mean', split_probe=None, copy_signal=True):
    """Removes noise by Common Average or Median Reference.
    Parameters
    ----------
    anas : np.array
           2d array of analog signals
    channels : list
               list of good channels to perform CAR/CMR with
    car_type : string
               'mean' or 'median'
    split_probe : int
                  splits anas into different probes to apply
                  car/cmr to each probe separately
    Returns
    -------
    anas_car : cleaned analog signals
    avg_ref : reference removed from signals
    """
    from copy import copy
    if channels is None:
        channels = np.arange(anas.shape[0])
    if copy_signal:
        anas_car = copy(anas)
    else:
        anas_car = anas
    anas_car = np.array(anas_car, dtype=np.float32)

    if car_type is 'mean':
        print('Applying CAR')
        if split_probe is not None:
            avg_ref = np.mean(anas_car[:split_probe], axis=0)
            anas_car[:split_probe] -= avg_ref
            avg_ref = np.mean(anas_car[split_probe:], axis=0)
            anas_car[split_probe:] -= avg_ref
        else:
            avg_ref = np.mean(anas_car[channels], axis=0)
            anas_car[channels] -= avg_ref
    elif car_type is 'median':
        print('Applying CMR')
        if split_probe is not None:
            avg_ref_1 = np.median(anas_car[:split_probe], axis=0)
            anas_car[:split_probe] -= avg_ref_1
            avg_ref_2 = np.median(anas_car[split_probe:], axis=0)
            anas_car[split_probe:] -= avg_ref_2
            avg_ref = np.array([avg_ref_1, avg_ref_2])
        else:
            avg_ref = np.median(anas_car[channels], axis=0)
            anas_car[channels] -= avg_ref
    else:
        raise AttributeError("'type must be 'mean' or 'median'")

    return anas_car, avg_ref

def extract_rising_edges(adc_signal, times, thresh=1.65):
    """Extract rising times from analog signal used as TTL.
    Parameters
    ----------
    adc_signal : np.array
                 1d array of analog TTL signal
    times : np.array
            timestamps array
    thresh: float
            threshold to detect 'high' value
    Returns
    -------
    rising_times : np.array with rising times
    """
    idx_high = np.where(adc_signal>1.65)[0]
    rising = []

    if len(idx_high) != 0:
        for i, idx in enumerate(idx_high[:-1]):
            if i==0:
                # first idx is rising
                rising.append(idx)
            elif idx - 1 != idx_high[i-1]:
                rising.append(idx)
    rising_times = times[rising]

    return rising_times

def filter_analog_signals(anas, freq, fs, filter_type='bandpass', filter_function='filtfilt', order=3, copy_signal=False):
    """Filters analog signals with zero-phase Butterworth filter.
    The function raises an Exception if the required filter is not stable.
    Parameters
    ----------
    anas : np.array
           2d array of analog signals
    freq : list or float
           cutoff frequency-ies in Hz
    fs : float
         sampling frequency
    filter_type : string
                  'lowpass', 'highpass', 'bandpass', 'bandstop'
    filter_function : string
        'filtfilt' or 'lfilter'
    order : int
            filter order
    parallel : bool
               parallel or not
    nprocesses : int
                 if parallel, number of processes
    Returns
    -------
    anas_filt : filtered signals
    """
    from scipy.signal import butter, filtfilt, lfilter
    fn = fs / 2.
    band = np.array(freq) / fn

    if filter_function == 'filtfilt':
        filterfun = filtfilt
    elif filter_function == 'lfilter':
        filterfun = lfilter
    else:
        raise NotImplementedError('filter-function {} not recognized'.format(filter_function))

    b, a = butter(order, band, btype=filter_type)

    if np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1):
        print('Filtering signals using', filter_function, 'with order', order, filter_type, 'with critical frequencies', freq ,'...')
        if len(anas.shape) == 2:
            anas_filt = filterfun(b, a, anas, axis=1)
        elif len(anas.shape) == 1:
            anas_filt = filterfun(b, a, anas)
        return anas_filt
    else:
        raise ValueError('Filter is not stable')

def ground_bad_channels(anas, bad_channels, copy_signal=True):
    """Grounds selected noisy channels.
    Parameters
    ----------
    anas : np.array
           2d array of analog signals
    bad_channels : list
                   list of channels to be grounded
    copy_signal : bool
                  copy signals or not
    Returns
    -------
    anas_zeros : analog signals with grounded channels
    """
    print('Grounding channels: ', bad_channels, '...')

    from copy import copy
    nsamples = anas.shape[1]
    if copy_signal:
        anas_zeros = copy(anas)
    else:
        anas_zeros = anas
    if type(bad_channels) is not list:
        bad_channels = [bad_channels]

    for i, ana in enumerate(anas_zeros):
        if i in bad_channels:
            anas_zeros[i] = np.zeros(nsamples)

    return anas_zeros


def duplicate_bad_channels(anas, bad_channels, probefile, copy_signal=True):
    """Duplicate selected noisy channels with channels in
    the same channel group.
    Parameters
    ----------
    anas : np.array
           2d array of analog signals
    bad_channels : list
                   list of channels to be grounded
    probefile : string
                absolute path to klusta-like probe file
    copy_signal : bool
                  copy signals or not
    Returns
    -------
    anas_dup : analog signals with duplicated channels
    """
    print('Duplicating good channels on channels: ', bad_channels, '...')

    def _select_rnd_chan_in_group(channel_map, ch_idx):
        for group_idx, group in channel_map.items():
            if ch_idx in group['channels']:
                gr = np.array(group['channels'])
                rnd_idx = np.random.choice(gr[gr != ch_idx])
                return rnd_idx

    probefile_ch_mapping = read_python(probefile)['channel_groups']

    from copy import copy
    nsamples = anas.shape[1]
    if copy_signal:
        anas_dup = copy(anas)
    else:
        anas_dup = anas
    if type(bad_channels) is not list:
        bad_channels = [bad_channels]

    for i, ana in enumerate(anas_dup):
        if i in bad_channels:
            rnd = _select_rnd_chan_in_group(probefile_ch_mapping, i)
            anas_dup[i] = anas[rnd]

    return anas_dup


def save_binary_format(filename, signal, spikesorter='klusta'):
    """Saves analog signals into klusta (time x chan) or spyking
    circus (chan x time) binary format (.dat)
    Parameters
    ----------
    filename : string
               absolute path (_klusta.dat or _spycircus.dat are appended)
    signal : np.array
             2d array of analog signals
    spikesorter : string
                  'klusta' or 'spykingcircus'
    Returns
    -------
    """
    if spikesorter is 'klusta':
        fdat = filename + '_klusta.dat'
        print('Saving ', fdat)
        with open(fdat, 'wb') as f:
            np.transpose(np.array(signal, dtype='float32')).tofile(f)
    elif spikesorter is 'spykingcircus':
        fdat = filename + '_spycircus.dat'
        print('Saving ', fdat)
        with open(fdat, 'wb') as f:
            np.array(signal, dtype='float32').tofile(f)


def create_klusta_prm(prb_path, nchan=32, fs=30000,
                      klusta_filter=True, filter_low=300, filter_high=6000,
                      filter_order=3,
                      use_single_threshold=True,
                      threshold_strong_std_factor=4.5,
                      threshold_weak_std_factor=2,
                      ncpus=8):
    """Creates klusta .prm files, with spikesorting parameters
    Parameters
    ----------
    prbpath : np.array
              2d array of analog signals
    nchan : int
            number of channels
    fs: float
        sampling frequency
    klusta_filter : bool
        filter with klusta or not
    filter_low: float
                low cutoff frequency (if klusta_filter is True)
    filter_high : float
                  high cutoff frequency (if klusta_filter is True)
    filter_order : int
        Butterworth filter order, default is 3.
    Returns
    -------
    full_filename : absolute path of .prm file
    """
    prb_path = op.abspath(prb_path)
    full_filename = prb_path + '.prm'
    print('Saving ', full_filename)
    with open(full_filename, 'w') as f:
        f.write('\n')
        f.write('experiment_name = ' + "r'" + prb_path + '_klusta' + "'" + '\n')
        f.write('prb_file = ' + "r'" + prb_path + ".prb'")
        f.write('\n')
        f.write('\n')
        f.write("traces = dict(\n\traw_data_files=[experiment_name + '.dat'],\n\tvoltage_gain=1.,"
                "\n\tsample_rate="+str(fs)+",\n\tn_channels="+str(nchan)+",\n\tdtype='float32',\n)")
        f.write('\n')
        f.write('\n')
        f.write("spikedetekt = dict(")
        if klusta_filter:
            f.write("\n\tfilter_low="+str(filter_low)+",\n\tfilter_high="+str(filter_high)+","
                    "\n\tfilter_butter_order="+str(filter_order)+",\n\tfilter_lfp_low=0,\n\tfilter_lfp_high=300,\n")
        f.write("\n\tchunk_size_seconds=1,\n\tchunk_overlap_seconds=.015,\n"
                "\n\tn_excerpts=50,\n\texcerpt_size_seconds=1,"
                "\n\tuse_single_threshold=" + str(use_single_threshold) +","
                "\n\tthreshold_strong_std_factor=" + str(threshold_strong_std_factor) + ",\n"
                "\tthreshold_weak_std_factor=" + str(threshold_weak_std_factor) + ",\n"
                "\tdetect_spikes='negative',"
                "\n\n\tconnected_component_join_size=1,\n"
                "\n\textract_s_before=16,\n\textract_s_after=64,\n"
                "\n\tn_features_per_channel=3,\n\tpca_n_waveforms_max=10000,\n)")
        f.write('\n')
        f.write('\n')
        f.write("klustakwik2 = dict(\n\tnum_starting_clusters=50,\n\tnum_cpus={}\n)".format(ncpus))
                # "\n\tnum_cpus=4,)")
    return full_filename


def find_frequency_range(anas, fs, freq_range, nchunks=30, chunksize=1*pq.s):
    """Finds a peak in the spectrum withink defined frequency range.
    Specta are computed on chunks of signals and averaged over channels.
    Parameters
    ----------
    anas : np.array
           2d array of stimulation analog signals
    fs : quantity
         sampling frequency in Hz
    freq_range: list/numpy array
                freq boundaries to find peak in
    nchunks: int
             number of chunks used to compute spectra
    chunksize: time Quantity
               length of chunks
    Returns
    -------
    fpeak : peak in Hz
    """
    from scipy import signal
    samples_per_chunk = int(fs.rescale('Hz').magnitude*chunksize.rescale('s').magnitude)
    random_intervals = np.random.randint(anas.shape[1]-samples_per_chunk, size=nchunks)
    samples = []
    for r in random_intervals:
        samples = np.concatenate((samples, np.arange(r, r+samples_per_chunk, dtype=int))).astype(int)
    anas_chunks = anas[:, samples]
    fpre, Pxxpre = signal.welch(anas_chunks, fs, nperseg=1024)
    avg_spectrum = np.mean(Pxxpre, axis=0)
    fpeak = fpre[np.where((fpre>freq_range[0]) &
                          (fpre<freq_range[1]))][np.argmax(avg_spectrum[np.where(
                              (fpre>freq_range[0]) &
                              (fpre<freq_range[1]))])]

    return int(fpeak)*pq.Hz


def save_geom_from_probefile(probefile, inter_group_dist=100,
                             inter_electrode_distance=10):
    prb_dict = read_python(probefile)
    positions = []
    for grp_id, electrodes in prb_dict['channel_groups'].items():
        for i_e, elec in enumerate(electrodes['channels']):
            pos = [0, (grp_id*inter_electrode_distance) + (i_e*inter_electrode_distance)]
            positions.append(pos)
    print(positions)

if __name__ == '__main__':
    prbpath = '/home/mikkel/apps/expipe-project/septum-mec/septum_mec/probes/tetrodes32ch-klusta-oe.prb'
    # save_geom_from_probefile(prbpath)
