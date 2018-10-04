print('Applying Mountainsort algorithm')

import mlpy
from shutil import copyfile

t_start = time.time()

if not os.path.isdir(join(self.rec_folder, 'mountain')):
    os.makedirs(join(self.rec_folder, 'mountain'))
self.mountain_folder = join(self.rec_folder, 'mountain')
rec_name = os.path.split(self.rec_folder)
if rec_name[-1] == '':
    rec_name = os.path.split(rec_name[0])[-1]
else:
    rec_name = rec_name[-1]
self.mountain_full_path = join(self.mountain_folder, rec_name)

# write data file
filename = join(self.mountain_folder, 'raw.mda')
mlpy.writemda32(self.recordings, filename)
print('saving ', filename)
radius = 50

# write csv probe file
with open(join(self.mountain_folder, 'geom.csv'), 'w') as f:
    for pos in self.mea_pos:
        f.write(str(pos[1]))
        f.write(',')
        f.write(str(pos[2]))
        f.write('\n')

# write param file
detect_threshold = None
params = {'samplerate': int(self.fs.rescale('Hz').magnitude), 'detect_sign': -1,
          "adjacency_radius": radius}
with open(join(self.mountain_folder, 'params.json'), 'w') as f:
    json.dump(params, f)

# copy mountainsort3.mlp
copyfile(join(self.root, 'spikesorter_files', 'mountainsort_files', 'mountainsort3.mlp'),
         join(self.mountain_folder, 'mountainsort3.mlp'))

if self.run_ss:
    print('Running MountainSort')
    self.curate=True

    import subprocess
    os.chdir(self.mountain_folder)
    try:
        t_start_proc = time.time()
        if self.curate:
            subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                     '--geom=geom.csv', '--firings_out=firings.mda', '--_params=params.json',
                                     '--curate=true'])
        else:
            subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                     '--geom=geom.csv', '--firings_out=firings.mda', '--_params=params.json'])
        self.processing_time = time.time() - t_start_proc
        print('Elapsed time: ', self.processing_time)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output)

    os.chdir(self.root)

    print('Parsing output files...')
    self.firings = mlpy.readmda(join(self.mountain_folder, 'firings.mda'))
    self.spike_trains = []
    clust_id, n_counts = np.unique(self.firings[2], return_counts=True)
    self.ml_times = self.times[self.firings[1].astype(int)]

    self.counts = 0
    for clust, count in zip(clust_id, n_counts):
        if count > self.minimum_spikes_per_cluster:
            idx = np.where(self.firings[2] == clust)[0]
            self.counts += len(idx)
            spike_times = self.ml_times[idx]
            spiketrain = neo.SpikeTrain(spike_times, t_start=self.t_start, t_stop=self.t_stop)
            self.spike_trains.append(spiketrain)
