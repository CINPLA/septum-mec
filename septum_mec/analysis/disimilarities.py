import expipe
from . import data_processing as dp
from .plotting import plot_waveforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def abs_diff(case, control,  statistic=None, axis=0):
    statistic = statistic or np.mean
    return np.abs(statistic(case, axis=axis) - statistic(control, axis=axis))


def disimilarity(s1, s2, statistic=None):
    statistic = statistic or np.mean
    w1 = s1.waveforms.magnitude
    w2 = s2.waveforms.magnitude
    ds = []
    nrc = w1.shape[1]
    for ch in range(nrc):
        d = abs_diff(w1[:, ch, :], w2[:, ch, :], statistic=statistic)
        ds.append(np.mean(d))
    return ds


_name = lambda u: u.annotations['name']


class Data:
    def __init__(self, project_path, action_ids, group_ids, position_sampling_rate, position_low_pass_frequency):
        project = expipe.get_project(project_path)
        actions = project.actions
        self._data = {}
        self.group_ids = group_ids
        self.action_ids = action_ids
        for action_id in action_ids:
            action  = actions[action_id]
            self._data[action_id] = {}
            data_path = dp.get_data_path(action)
            epochs = dp.load_epochs(data_path)
            x, y, t, speed = dp.load_tracking(
                data_path, position_sampling_rate, position_low_pass_frequency)
            a, at = dp.load_head_direction(
                data_path, position_sampling_rate, position_low_pass_frequency)

            self._data[action_id] = {
                'x': x,
                'y': y,
                't': t,
                'speed': speed,
                'a': a,
                'at': at,
                'epochs': epochs,
                'groups': {}
            }

            for group_id in group_ids:
                try:
                    anas = dp.load_lfp(data_path, group_id) # TODO remove artifacts
                except:
                    print(group_id)
                    raise
                spike_trains = dp.load_spiketrains(data_path, group_id, load_waveforms=True)

                self._data[action_id]['groups'][group_id] = {
                    'spike_trains': {_name(s): s for s in dp.sort_by_cluster_id(spike_trains)},
                    'anas': anas
                }

    def __getitem__(self, action_id):
        if isinstance(action_id, int):
            return self._data[self.action_ids[action_id]]
        else:
            return self._data[action_id]

    def get_unit(self, name, action_id):
        sptr = [
            u for g in self[action_id]['groups'].values()
            for u in g['spike_trains'].values()
            if _name(u)==name
        ]
        if len(sptr) == 1:
            return sptr[0]
        elif len(sptr) == 0:
            raise ValueError('Unable to locate {}'.format(name))
        else:
            raise IOError('Multiple units (n {}) with name {}'.format(len(sptr), name))

    def get_units(self, action_id):
        return [
            u for g in self[action_id]['groups'].values()
            for u in g['spike_trains'].values()
        ]

    def calculate_dissimilarities(self, threshold=6, plot_comparison=True):
        disimilarities = {}
        for group_id, values1 in self[0]['groups'].items():
            for n1, s1 in values1['spike_trains'].items():
                disimilarities[n1] = {}
                for n2, s2 in self[1]['groups'][group_id]['spike_trains'].items():
                    ds = disimilarity(s1, s2, np.median)
                    disimilarities[n1][n2] = np.mean(ds)
        disimilarities = pd.DataFrame(disimilarities).T
        disimilarities = disimilarities[disimilarities < threshold]

        idxmin = disimilarities.idxmin(axis=1)
        minvals = disimilarities.min(axis=1)
        pairs = np.vstack([idxmin.index, idxmin.values, minvals]).T
        pairs = pairs[~np.isnan(pairs[:,2].astype(float)),:]

        # select min disimilarities in repeated neurons from session 2
        n, c = np.unique(pairs[:,1], return_counts=True)
        repeats = n[c > 1]

        delete_idxs = []
        for r in repeats:
            m, = np.where(pairs[:,1] == r)
            idx = m[np.argmin(pairs[m, 2])]
            delete_idxs.extend(m[m!=idx].tolist())

        pairs = np.delete(pairs, delete_idxs, axis=0)

        self._pairs = pairs

        remains = {}

        for i, action_id in enumerate(self.action_ids):
            remains[action_id] = [
                _name(u) for u in self.get_units(action_id)
                if _name(u) not in pairs[:,i]]

        self._remains = remains

        for pair in pairs:
            fig = plt.figure()
            plt.suptitle('{} vs {}, disimilarity = {}'.format(*pair, disimilarities.loc[pair[0], pair[1]]))
            plot_waveforms(self.get_unit(pair[0], 0), fig=fig, f=np.median, color='r')
            plot_waveforms(self.get_unit(pair[1], 1), fig=fig, f=np.median, color='k', linestyle='--')

    @property
    def pairs(self):
        if hasattr(self, '_pairs'):
            return self._pairs
        else:
            self.calculate_dissimilarities(plot_comparison=False)
            return self._pairs

    @property
    def remains(self):
        if hasattr(self, '_remains'):
            return self._remains
        else:
            self.calculate_dissimilarities(plot_comparison=False)
            return self._remains
