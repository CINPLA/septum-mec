# This is work in progress,
import neo
import numpy as np
import exdir
import exdir.plugins.quantities
import exdir.plugins.git_lfs
import pathlib
import os
import quantities as pq
import spikeextractors as se
import expipe
import spatial_maps as sp
import warnings


def project_path():
    path = os.environ.get("SEPTUM_MEC_DATA")
    if path is None:
        raise Exception("Need to set `SEPTUM_MEC_DATA` as environment variable first.")
    else:
        path = pathlib.Path(path)
    return path


def view_active_channels(action, sorter):
    path = get_data_path(action)
    elphys_path = path / 'processing' / 'electrophysiology'
    sorter_path = elphys_path / 'spikesorting' / sorter / 'phy'
    return np.load(sorter_path / 'channel_map_si.npy')


def _cut_to_same_len(*args):
    out = []
    lens = []
    for arg in args:
        lens.append(len(arg))
    minlen = min(lens)
    for arg in args:
        out.append(arg[:minlen])
    return out


def _read_epoch(exdir_file, path, lazy=False):
    group = exdir_file[path]
    if lazy:
        times = []
    else:
        times = pq.Quantity(group['timestamps'].data,
                            group['timestamps'].attrs['unit'])

    if "durations" in group and not lazy:
        durations = pq.Quantity(group['durations'].data, group['durations'].attrs['unit'])
    elif "durations" in group and lazy:
        durations = []
    else:
        durations = None

    if 'data' in group and not lazy:
        if 'unit' not in group['data'].attrs:
            labels = group['data'].data
        else:
            labels = pq.Quantity(group['data'].data,
                                 group['data'].attrs['unit'])
    elif 'data' in group and lazy:
        labels = []
    else:
        labels = None
    annotations = {'exdir_path': path}
    annotations.update(group.attrs.to_dict())

    if lazy:
        lazy_shape = (group.attrs['num_samples'],)
    else:
        lazy_shape = None
    epo = neo.Epoch(times=times, durations=durations, labels=labels,
                lazy_shape=lazy_shape, **annotations)

    return epo


def velocity_filter(x, y, t, threshold):
    """
    Removes values above threshold
    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    threshold : float
    """
    assert len(x) == len(y) == len(t), 'x, y, t must have same length'
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)

    vel = np.array([dx, dy]) / dt
    speed = np.linalg.norm(vel, axis=0)
    speed_mask = (speed < threshold)
    speed_mask = np.append(speed_mask, 0)
    x = x[np.where(speed_mask)]
    y = y[np.where(speed_mask)]
    t = t[np.where(speed_mask)]
    return x, y, t


def interp_filt_position(x, y, tm, fs=100 , f_cut=10 ):
    """
    rapid head movements will contribute to velocity artifacts,
    these can be removed by low-pass filtering
    see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    code addapted from Espen Hagen
    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    tm : quantities.Quantity array in s
        1d vector of times at x, y positions
    fs : quantities scalar in Hz
        return radians
    Returns
    -------
    out : angles, resized t
    """
    import scipy.signal as ss
    assert len(x) == len(y) == len(tm), 'x, y, t must have same length'
    t = np.arange(tm.min(), tm.max() + 1. / fs, 1. / fs)
    x = np.interp(t, tm, x)
    y = np.interp(t, tm, y)
    # rapid head movements will contribute to velocity artifacts,
    # these can be removed by low-pass filteringpar
    # see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    # code addapted from Espen Hagen
    b, a = ss.butter(N=1, Wn=f_cut * 2 / fs)
    # zero phase shift filter
    x = ss.filtfilt(b, a, x)
    y = ss.filtfilt(b, a, y)
    # we tolerate small interpolation errors
    x[(x > -1e-3) & (x < 0.0)] = 0.0
    y[(y > -1e-3) & (y < 0.0)] = 0.0

    return x, y, t


def rm_nans(*args):
    """
    Removes nan from all corresponding arrays
    Parameters
    ----------
    args : arrays, lists or quantities which should have removed nans in
           all the same indices
    Returns
    -------
    out : args with removed nans
    """
    nan_indices = []
    for arg in args:
        nan_indices.extend(np.where(np.isnan(arg))[0].tolist())
    nan_indices = np.unique(nan_indices)
    out = []
    for arg in args:
        out.append(np.delete(arg, nan_indices))
    return out


def unit_path(channel_id, unit_id):
    return "/processing/electrophysiology/channel_group_{}/UnitTimes/{}".format(channel_id, unit_id)


def load_leds(data_path):
    root_group = exdir.File(
        data_path, "r",
        plugins=[exdir.plugins.quantities, exdir.plugins.git_lfs])

    # tracking data
    position_group = root_group['processing']['tracking']['camera_0']['Position']
    stop_time = position_group.attrs["stop_time"]
    print(position_group['led_0']['data'].directory)
    x1, y1 = position_group['led_0']['data'].data.T
    t1 = position_group['led_0']['timestamps'].data
    x2, y2 = position_group['led_1']['data'].data.T
    t2 = position_group['led_1']['timestamps'].data

    return x1, y1, t1, x2, y2, t2, stop_time


def filter_xy_zero(x, y, t):
    idxs, = np.where((x == 0) & (y == 0))
    return [np.delete(a, idxs) for a in [x, y, t]]

def filter_xy_box_size(x, y, t, box_size):
    idxs, = np.where((x > box_size[0]) | (x < 0) | (y > box_size[1]) | (y < 0))
    return [np.delete(a, idxs) for a in [x, y, t]]


def filter_t_zero_duration(x, y, t, duration):
    idxs, = np.where((t < 0) | (t > duration))
    return [np.delete(a, idxs) for a in [x, y, t]]


def load_head_direction(data_path, sampling_rate, low_pass_frequency, box_size):
    from head_direction.head import head_direction
    x1, y1, t1, x2, y2, t2, stop_time = load_leds(data_path)

    x1, y1, t1 = rm_nans(x1, y1, t1)
    x2, y2, t2 = rm_nans(x2, y2, t2)

    x1, y1, t1 = filter_t_zero_duration(x1, y1, t1, stop_time.magnitude)
    x2, y2, t2 = filter_t_zero_duration(x2, y2, t2, stop_time.magnitude)

    # OE saves 0.0 when signal is lost, these can be removed
    x1, y1, t1 = filter_xy_zero(x1, y1, t1)
    x2, y2, t2 = filter_xy_zero(x2, y2, t2)

    # x1, y1, t1 = filter_xy_box_size(x1, y1, t1, box_size)
    # x2, y2, t2 = filter_xy_box_size(x2, y2, t2, box_size)

    x1, y1, t1 = interp_filt_position(x1, y1, t1,
        fs=sampling_rate, f_cut=low_pass_frequency)
    x2, y2, t2 = interp_filt_position(x2, y2, t2,
        fs=sampling_rate, f_cut=low_pass_frequency)

    x1, y1, t1, x2, y2, t2 = _cut_to_same_len(x1, y1, t1, x2, y2, t2)

    check_valid_tracking(x1, y1, box_size)
    check_valid_tracking(x2, y2, box_size)

    angles, times = head_direction(x1, y1, x2, y2, t1)
    return angles, times


def check_valid_tracking(x, y, box_size):
    if np.isnan(x).any() and np.isnan(y).any():
        raise ValueError('nans found in  position, ' +
            'x nans = %i, y nans = %i' % (sum(np.isnan(x)), sum(np.isnan(y))))

    if (x.min() < 0 or x.max() > box_size[0] or y.min() < 0 or y.max() > box_size[1]):
        warnings.warn(
            "Invalid values found " +
            "outside box: min [x, y] = [{}, {}], ".format(x.min(), y.min()) +
            "max [x, y] = [{}, {}]".format(x.max(), y.max()))


def load_tracking(data_path, sampling_rate, low_pass_frequency, box_size, velocity_threshold=5):
    x1, y1, t1, x2, y2, t2, stop_time = load_leds(data_path)
    x1, y1, t1 = rm_nans(x1, y1, t1)
    x2, y2, t2 = rm_nans(x2, y2, t2)

    x1, y1, t1 = filter_t_zero_duration(x1, y1, t1, stop_time.magnitude)
    x2, y2, t2 = filter_t_zero_duration(x2, y2, t2, stop_time.magnitude)

    # select data with least nan
    if len(x1) > len(x2):
        x, y, t = x1, y1, t1
    else:
        x, y, t = x2, y2, t2

    # OE saves 0.0 when signal is lost, these can be removed
    x, y, t = filter_xy_zero(x, y, t)

    # x, y, t = filter_xy_box_size(x, y, t, box_size)

    # remove velocity artifacts
    x, y, t = velocity_filter(x, y, t, velocity_threshold)

    x, y, t = interp_filt_position(
        x, y, t, fs=sampling_rate, f_cut=low_pass_frequency)

    check_valid_tracking(x, y, box_size)

    vel = np.gradient([x, y], axis=1) / np.gradient(t)
    speed = np.linalg.norm(vel, axis=0)

    return x, y, t, speed


def get_data_path(action):
    # action_path = action._backend.path
    # project_path = action_path.parent.parent
    #data_path = action.data['main']
    data_path = str(action.data_path('main'))

    # print("Project path: {}\nData path: {}".format(project_path, data_path))
    return data_path


def get_sample_rate(data_path, default_sample_rate=30000*pq.Hz):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    sr = default_sample_rate
    if 'processing' in f.keys():
        processing = f['processing']
        if 'electrophysiology' in processing.keys():
            ephys = processing['electrophysiology']
            if 'sample_rate' in ephys.attrs.keys():
                sr = ephys.attrs['sample_rate']
    return sr


def get_duration(data_path):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])

    return f.attrs['session_duration'].rescale('s')


def load_lfp(data_path, channel_group):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    # LFP
    t_stop = f.attrs['session_duration']
    _lfp = f['processing']['electrophysiology']['channel_group_{}'.format(channel_group)]['LFP']
    keys = list(_lfp.keys())
    electrode_value = [_lfp[key]['data'].value.flatten() for key in keys]
    electrode_idx = [_lfp[key].attrs['electrode_idx'] for key in keys]
    sampling_rate = _lfp[keys[0]].attrs['sample_rate']
    units = _lfp[keys[0]]['data'].attrs['unit']
    LFP = np.r_[[_lfp[key]['data'].value.flatten() for key in keys]].T
    LFP = LFP[:, np.argsort(electrode_idx)]

    LFP = neo.AnalogSignal(
        LFP, units=units, t_stop=t_stop, sampling_rate=sampling_rate,
        **{'electrode_idx': electrode_idx})
    LFP = LFP.rescale('mV')
    return LFP


def sort_by_cluster_id(spike_trains):
    if len(spike_trains) == 0:
        return spike_trains
    if 'name' not in spike_trains[0].annotations:
        print('Unable to get cluster_id, save with phy to create')
    sorted_sptrs = sorted(
        spike_trains,
        key=lambda x: int(x.annotations['name'].lower().replace('unit #', '')))
    return sorted_sptrs


def load_epochs(data_path):
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    epochs_group = f['epochs']
    epochs = []
    for group in epochs_group.values():
        if 'timestamps' in group.keys():
            epo = _read_epoch(f, group.name)
            epochs.append(epo)
        else:
            for g in group.values():
                if 'timestamps' in g.keys():
                    epo = _read_epoch(f, g.name)
                    epochs.append(epo)
    return epochs


def get_channel_groups(data_path):
    '''
    Returns channeø groups of processing/electrophysiology
    Parameters
    ----------
    data_path: Path
        The action data path
    Returns
    -------
    channel groups: list
        The channel groups
    '''
    f = exdir.File(str(data_path), 'r', plugins=[exdir.plugins.quantities])
    channel_groups = []
    if 'processing' in f.keys():
        processing = f['processing']
        if 'electrophysiology' in processing.keys():
            ephys = processing['electrophysiology']
            for chname, ch in ephys.items():
                if 'channel' in chname:
                    channel_groups.append(int(chname.split('_')[-1]))
    return channel_groups


def get_unit_id(unit):
    try:
        uid = int(unit.annotations['name'].split('#')[-1])
    except AttributeError:
        uid = int(unit['name'].split('#')[-1])
    return uid


def load_spiketrains(data_path, channel_group=None, load_waveforms=False, t_start=0 * pq.s):
    '''
    Parameters
    ----------
    data_path
    channel_group
    load_waveforms
    remove_label
    Returns
    -------
    '''
    sample_rate = get_sample_rate(data_path)
    sorting = se.ExdirSortingExtractor(
        data_path, sample_rate=sample_rate,
        channel_group=channel_group, load_waveforms=load_waveforms)
    sptr = []
    # build neo pbjects
    for u in sorting.get_unit_ids():
        times = sorting.get_unit_spike_train(u) / sample_rate
        t_stop = get_duration(data_path)
        times = times - t_start
        t_stop = t_stop - t_start
        times = times[(times > 0) & (times <= t_stop)]
        if load_waveforms and 'waveforms' in sorting.get_unit_spike_feature_names(u):
            wf = sorting.get_unit_spike_features(u, 'waveforms')
            wf = wf[np.where(times > 0)] * pq.uV
        else:
            wf = None
        st = neo.SpikeTrain(times=times, t_stop=t_stop, waveforms=wf, sampling_rate=sample_rate)
        for p in sorting.get_unit_property_names(u):
            st.annotations.update({p: sorting.get_unit_property(u, p)})
        sptr.append(st)

    return sptr


def load_unit_annotations(data_path, channel_group):
    '''
    Parameters
    ----------
    data_path
    channel_group
    Returns
    -------
    '''
    sample_rate = get_sample_rate(data_path)
    sorting = se.ExdirSortingExtractor(
        data_path, sample_rate=sample_rate,
        channel_group=channel_group, load_waveforms=False)
    units = []
    for u in sorting.get_unit_ids():
        annotations = {}
        for p in sorting.get_unit_property_names(u):
            annotations.update({p: sorting.get_unit_property(u, p)})
        units.append(annotations)
    return units


def load_spike_train(data_path, channel_id, unit_id, stop_time=None):
    root_group = exdir.File(data_path, "r", plugins=[exdir.plugins.quantities,
                                                exdir.plugins.git_lfs])
    u_path = unit_path(channel_id, unit_id)
    unit_group = root_group[u_path]
    # spiketrain data
    sptr_group = unit_group
    metadata = {}
    times = np.array(sptr_group['times'].data)
    if stop_time is not None:
        t_stop = stop_time
        times = times[times < t_stop]
    t_start = 0 * pq.s
    metadata.update(sptr_group['times'].attrs.to_dict())
    metadata.update({'exdir_path': str(data_path)})
    sptr = neo.SpikeTrain(times=times, units = 's',
                      t_stop=t_stop,
                      t_start=t_start,
                      waveforms=None,
                      sampling_rate=None,
                      **metadata)
    return sptr


class Template:
    def __init__(self, sptr):
        self.data = np.array(sptr.waveforms.mean(0))
        self.sampling_rate = float(sptr.sampling_rate)


class Data:
    def __init__(self, **kwargs):
        self.project_path = project_path()
        self.params = kwargs
        self.project = expipe.get_project(self.project_path)
        self.actions = self.project.actions
        self._spike_trains = {}
        self._templates = {}
        self._stim_times = {}
        self._tracking = {}
        self._head_direction = {}
        self._lfp = {}
        self._occupancy = {}
        self._prob_dist = {}
        self._spatial_bins = None

    def data_path(self, action_id):
        return pathlib.Path(self.project_path) / "actions" / action_id / "data" / "main.exdir"

    def duration(self, action_id):
        return get_duration(self.data_path(action_id))

    def tracking(self, action_id):
        if action_id not in self._tracking:
            x, y, t, speed = load_tracking(
                self.data_path(action_id),
                sampling_rate=self.params['position_sampling_rate'],
                low_pass_frequency=self.params['position_low_pass_frequency'],
                box_size=self.params['box_size'])
            self._tracking[action_id] = {
                'x': x, 'y': y, 't': t, 'v': speed
            }
        return self._tracking[action_id]

    @property
    def spatial_bins(self):
        if self._spatial_bins is None:
            box_size_, bin_size_ = sp.maps._adjust_bin_size(
                box_size=self.params['box_size'],
                bin_size=self.params['bin_size'])
            xbins, ybins = sp.maps._make_bins(box_size_, bin_size_)
            self._spatial_bins = (xbins, ybins)
            self.box_size_, self.bin_size_ = box_size_, bin_size_
        return self._spatial_bins

    def occupancy(self, action_id):
        if action_id not in self._occupancy:
            xbins, ybins = self.spatial_bins

            occupancy_map = sp.maps._occupancy_map(
                self.tracking(action_id)['x'],
                self.tracking(action_id)['y'],
                self.tracking(action_id)['t'], xbins, ybins)
            self._occupancy[action_id] = occupancy_map
        return self._occupancy[action_id]

    def prob_dist(self, action_id):
        if action_id not in self._prob_dist:
            xbins, ybins = xbins, ybins = self.spatial_bins
            prob_dist = sp.stats.prob_dist(
                self.tracking(action_id)['x'],
                self.tracking(action_id)['y'], bins=(xbins, ybins))
            self._prob_dist[action_id] = prob_dist
        return self._prob_dist[action_id]

    def head_direction(self, action_id):
        if action_id not in self._head_direction:
            a, t = load_head_direction(
                self.data_path(action_id),
                sampling_rate=self.params['position_sampling_rate'],
                low_pass_frequency=self.params['position_low_pass_frequency'],
                box_size=self.params['box_size'])
            self._head_direction[action_id] = {
                'a': a, 't': t
            }
        return self._head_direction[action_id]

    def lfp(self, action_id, channel_group):
        if action_id not in self._lfp:
            self._lfp[action_id] = load_lfp(
                self.data_path(action_id), channel_group) # TODO remove artifacts
        return self._lfp[action_id]

    def template(self, action_id, channel_group, unit_id):
        if action_id not in self._templates:
            self._templates[action_id] = {}
        if channel_group not in self._templates[action_id]:
            self._templates[action_id][channel_group] = {
                get_unit_id(st): Template(st)
                for st in load_spiketrains(
                    self.data_path(action_id), channel_group, load_waveforms=True)
            }
        return self._templates[action_id][channel_group][unit_id]

    def spike_train(self, action_id, channel_group, unit_id):
        if action_id not in self._spike_trains:
            self._spike_trains[action_id] = {}
        if channel_group not in self._spike_trains[action_id]:
            self._spike_trains[action_id][channel_group] = {
                get_unit_id(st): st
                for st in load_spiketrains(
                    self.data_path(action_id), channel_group, load_waveforms=False)
            }
        return self._spike_trains[action_id][channel_group][unit_id]

    def stim_times(self, action_id):
        if action_id not in self._stim_times:
            epochs = load_epochs(self.data_path(action_id))
            if len(epochs) == 0:
                self._stim_times[action_id] = None
            elif len(epochs) == 1:
                stim_times = epochs[0]
                stim_times = np.sort(np.abs(stim_times))
                # there are some 0 times and inf times, remove those
                stim_times = stim_times[stim_times <= self.duration(action_id)]
                stim_times = stim_times[stim_times >= 1e-20]
                self._stim_times[action_id] = stim_times
            else:
                raise ValueError('Found multiple epochs')
        return self._stim_times[action_id]
