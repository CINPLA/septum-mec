from septum_mec.imports import *
from expipe_plugin_cinpla.tools import action as action_tools
from septum_mec.analysis import signals as sig_tools
from datetime import timedelta
from expipe_plugin_cinpla.tools import config
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
                  required=True,
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

        #TODO preprocessing: cmr, grounding
        recording = si.OpenEphysRecordingExtractor(openephys_path)
        si.loadProbeFile(recording, probe_path)
        if sorter == 'klusta':
            sorting = st.sorSTRINGters.klusta(recording)
            #TODO multiprocessing
        elif sorter == 'mountain':
            sorting = st.sorters.mountainsort4(recording, adjacency_radius=10, detect_sign=-1)
        elif sorter == 'kilosort':
            sorting = st.sorters.kilosort(recording, kilosort_path='/home/mikkel/apps/KiloSort', npy_matlab_path='/home/mikkel/apps/npy-matlab/npy-matlab')
        st.exportToPhy(recording, sorting, openephys_path / 'phy')
            # sorting_cuated = si.PhysortingExtractor(openephys_path / 'phy')
        #TODO store data in exdir sorting extractor
        # si.ExdirSortingExtractor.writeSorting(sorting, exdir_path)
