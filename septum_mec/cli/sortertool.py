from septum_mec.imports import *
from expipe_plugin_cinpla.tools import action as action_tools
from septum_mec.analysis import signals as sig_tools
from datetime import timedelta
from expipe_plugin_cinpla.tools import config
import spikeextractors as se
import spiketoolkit as st
import os.path as op


def attach_to_cli(cli):
    @cli.command('process',
                 short_help='Generate a klusta .dat and .prm files from openephys directory.')
    @click.argument('action-id', type=click.STRING)
    @click.option('--probe-path',
                  type=click.STRING,
                  help='Path to probefile, assumed to be in expipe config directory by default.',
                  )
    @click.option('--openephys-path',
                  type=click.STRING,
                  help='Path to openeophys dir, if none it is deduced from action id.',
                  )
    @click.option('--exdir-path',
                  type=click.STRING,
                  help='Path to desired exdir directory, if none it is deduced from action id.',
                  )
    @click.option('--sorter',
                  default='klusta',
                  type=click.Choice(['klusta', 'mountain', 'kilosort']),
                  help='',
                  )
    def process_openephys(action_id, probe_path, exdir_path, openephys_path, sorter):
        import spikeinterface as si
        import spiketoolkit as st
        project = expipe.get_project(PAR.PROJECT_ROOT)
        action = project.actions[action_id]
        if exdir_path is None:
            exdir_path = action_tools._get_data_path(action)
            exdir_file = exdir.File(exdir_path, plugins=exdir.plugins.quantities)
        if openephys_path is None:
            acquisition = exdir_file["acquisition"]
            if acquisition.attrs['acquisition_system'] is None:
                raise ValueError('No Open Ephys aquisition system ' +
                                 'related to this action')
            openephys_session = acquisition.attrs["openephys_session"]
            openephys_path = os.path.join(str(acquisition.directory), openephys_session)
            probe_path = probe_path or project.config.get('probe')

        recording = se.OpenEphysRecordingExtractor(openephys_path)
        se.loadProbeFile(recording, probe_path)
        # apply cmr
        recording_cmr = st.preprocessing.common_reference(recording)
        recording_lfp = st.preprocessing.bandpass_filter(recording, freq_min=1, freq_max=300)
        recording_lfp = st.preprocessing.resample(recording, 1000)

        if sorter == 'klusta':
            sorting = st.sorters.klusta(recording, by_property='group')
        elif sorter == 'mountain':
            sorting = st.sorters.mountainsort4(recording, by_property='group',
                                               adjacency_radius=10, detect_sign=-1)
        elif sorter == 'kilosort':
            sorting = st.sorters.kilosort(recording, by_property='group',
                                          kilosort_path='/home/mikkel/apps/KiloSort',
                                          npy_matlab_path='/home/mikkel/apps/npy-matlab/npy-matlab')
        # extract waveforms

        se.ExdirSortingExtractor.writeSorting(sorting, exdir_path, sample_rate=recording.getSamplingFrequency())
