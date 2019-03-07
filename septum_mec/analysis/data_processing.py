# This is work in progress,
import neo
import numpy as np
import exdir.plugins.quantities
import exdir.plugins.git_lfs
import pathlib
import exdir
import os


def _cut_to_same_len(*args):
    out = []
    lens = []
    for arg in args:
        lens.append(len(arg))
    minlen = min(lens)
    for arg in args:
        out.append(arg[:minlen])
    return out


def project_path():
    result = pathlib.Path(os.environ.get("CHARLOTTE_PNN_MEC_DATA"))
    if result is None:
        raise Exception("Need to set `CHARLOTTE_PNN_MEC_DATA` as environment variable first.")
    return result


def fftcorrelate2d(arr1, arr2, mode='full', normalize=False):
    # TODO replace with astropy, just make sure the results are the same first
    from scipy.signal import fftconvolve
    if normalize:
        a_ = np.reshape(arr1, (1, arr1.size))
        v_ = np.reshape(arr2, (1, arr2.size))
        arr1 = (arr1 - np.mean(a_)) / (np.std(a_) * len(a_))
        arr2 = (arr2 - np.mean(v_)) / np.std(v_)
    corr = fftconvolve(arr1, np.fliplr(np.flipud(arr2)), mode=mode)
    return corr


def velocity_threshold(x, y, t, threshold):
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
    r = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    v = np.divide(r, np.diff(t))
    speed_lim = np.concatenate(([False], v > threshold), axis=0)
    x[speed_lim] = np.nan
    y[speed_lim] = np.nan
    x, y, t = rm_nans(x, y, t)
    return x, y, t

def select_best_position(x1, y1, t1, x2, y2, t2, speed_filter=5):
    """
    selects position data with least nan after speed filtering
    Parameters
    ----------
    x1 : quantities.Quantity array in m
        1d vector of x positions from LED 1
    y1 : quantities.Quantity array in m
        1d vector of x positions from LED 1
    t1 : quantities.Quantity array in s
        1d vector of times from LED 1 at x, y positions
    x2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    y2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    t2 : quantities.Quantity array in s
        1d vector of times from LED 2 at x, y positions
    speed_filter : None or quantities in m/s
        threshold filter for translational speed
    """
    x1, y1, t1, x2, y2, t2 = _cut_to_same_len(x1, y1, t1, x2, y2, t2)
    measurements1 = len(x1)
    measurements2 = len(x2)
    x1, y1, t1 = rm_nans(x1, y1, t1)
    x2, y2, t2 = rm_nans(x2, y2, t2)
    if speed_filter is not None:
        x1, y1, t1 = velocity_threshold(x1, y1, t1, speed_filter)
        x2, y2, t2 = velocity_threshold(x2, y2, t2, speed_filter)

    if len(x1) > len(x2):
        # print('Removed %.2f %% invalid measurements in path' %
              # ((1. - len(x1) / float(measurements1)) * 100.))
        x = x1
        y = y1
        t = t1
    else:
        # print('Removed %.2f %% invalid measurements in path' %
              # ((1. - len(x2) / float(measurements2)) * 100.))
        x = x2
        y = y2
        t = t2
    return x, y, t


def interp_filt_position(x, y, tm, box_xlen=1 , box_ylen=1 ,
                         pos_fs=100 , f_cut=10 ):
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
    pos_fs : quantities scalar in Hz
        return radians
    Returns
    -------
    out : angles, resized t
    """
    import scipy.signal as ss
    assert len(x) == len(y) == len(tm), 'x, y, t must have same length'
    t = np.arange(tm.min(), tm.max() + 1. / pos_fs, 1. / pos_fs)
    x = np.interp(t, tm, x)
    y = np.interp(t, tm, y)
    # rapid head movements will contribute to velocity artifacts,
    # these can be removed by low-pass filtering
    # see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    # code addapted from Espen Hagen
    b, a = ss.butter(N=1, Wn=f_cut * 2 / pos_fs)
    # zero phase shift filter
    x = ss.filtfilt(b, a, x)
    y = ss.filtfilt(b, a, y)
    # we tolerate small interpolation errors
    x[(x > -1e-3) & (x < 0.0)] = 0.0
    y[(y > -1e-3) & (y < 0.0)] = 0.0
    if np.isnan(x).any() and np.isnan(y).any():
        raise ValueError('nans found in  position, ' +
            'x nans = %i, y nans = %i' % (sum(np.isnan(x)), sum(np.isnan(y))))
    if (x.min() < 0 or x.max() > box_xlen or y.min() < 0 or y.max() > box_ylen):
        raise ValueError(
            "Interpolation produces path values " +
            "outside box: min [x, y] = [{}, {}], ".format(x.min(), y.min()) +
            "max [x, y] = [{}, {}]".format(x.max(), y.max()))

    R = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    V = R / np.diff(t)
    # print('Maximum speed {}'.format(V.max()))
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


def load_data(data_path, unit_path, par):
    raise DeprecationWarning("load_data is deprecated. Use load_tracking and load_spike_train instead.")
    x, y, t, speed = load_tracking(data_path, par)
    sptr = load_spike_train(data_path, unit_path, t)
    return x, y, t, speed, sptr


def load_leds(data_path, par):
    plugins = [exdir.plugins.quantities, exdir.plugins.git_lfs]
    root_group = exdir.File(data_path, "r", plugins=plugins)

    # tracking data
    position_group = root_group['processing']['tracking']['camera_0']['Position']
    x1, y1 = position_group['led_0']['data'].data.T
    t1 = position_group['led_0']['timestamps'].data
    x2, y2 = position_group['led_1']['data'].data.T
    t2 = position_group['led_1']['timestamps'].data

    nans = (np.isnan(x1) | np.isnan(x2) | np.isnan(y1) | np.isnan(y2))

    stop_time = position_group.attrs.to_dict()["stop_time"]
    time_mask = (t1 <= stop_time)

    mask = (time_mask & ~nans)

    x1 = x1[np.where(mask)]
    y1 = y1[np.where(mask)]
    t1 = t1[np.where(mask)]
    x2 = x2[np.where(mask)]
    y2 = y2[np.where(mask)]
    t2 = t2[np.where(mask)]

    dt = np.diff(t1)
    dx = np.diff(x1)
    dy = np.diff(y1)

    vel = np.array([dx, dy]) / dt
    speed = np.linalg.norm(vel, axis=0)
    speed_mask = (speed < 5)
    speed_mask = np.append(speed_mask, 0)

    x1 = x1[np.where(speed_mask)]
    y1 = y1[np.where(speed_mask)]
    t1 = t1[np.where(speed_mask)]
    x2 = x2[np.where(speed_mask)]
    y2 = y2[np.where(speed_mask)]
    t2 = t2[np.where(speed_mask)]

    return x1, y1, t1, x2, y2, t2


def load_tracking(data_path, par):
    root_group = exdir.File(data_path, "r", plugins=[exdir.plugins.quantities,
                                                exdir.plugins.git_lfs])

    # tracking data
    position_group = root_group['processing']['tracking']['camera_0']['Position']
    stop_time = position_group.attrs.to_dict()["stop_time"]
    x1, y1 = position_group['led_0']['data'].data.T
    t1 = position_group['led_0']['timestamps'].data
    x2, y2 = position_group['led_1']['data'].data.T
    t2 = position_group['led_1']['timestamps'].data

    x, y, t = select_best_position(x1, y1, t1, x2, y2, t2)
    x, y, t = interp_filt_position(x, y, t, pos_fs=par['pos_fs'], f_cut=par['f_cut'])
    mask = t <= stop_time
    #mask = t2 <= stop_time
    x = x[mask]
    y = y[mask]
    t = t[mask]

    dt = np.mean(np.diff(t))
    vel = np.gradient([x,y],axis=1)/dt
    speed = np.linalg.norm(vel,axis=0)

    return x, y, t, speed


def load_spike_train(data_path, unit_path, t):
    root_group = exdir.File(data_path, "r", plugins=[exdir.plugins.quantities,
                                                exdir.plugins.git_lfs])
    unit_group = root_group[unit_path]
    # spiketrain data
    sptr_group = unit_group
    metadata = {}
    times = np.array(sptr_group['times'].data)

    # Remove spikes that fall outside the range of times.
    # Otherwise, interpolation will fail when plotting spikes in scatter plot.
    times_count_before = len(times)
    times = times[(times > min(t)) & (times < max(t))]
    times_count_after = len(times)
    times_count_diff = times_count_before - times_count_after

    # if times_count_diff != 0:
        # print("Removed {} spikes that fell outside range of valid position data.".format(times_count_diff))

    t_stop = sptr_group.parent.attrs['stop_time']
    t_start = sptr_group.parent.attrs['start_time']
    metadata.update(sptr_group['times'].attrs.to_dict())
    metadata.update({'exdir_path': str(data_path)})
    sptr = neo.SpikeTrain(times=times, units = 's',
                      t_stop=t_stop,
                      t_start=t_start,
                      waveforms=None,
                      sampling_rate=None,
                      **metadata)
    return sptr

