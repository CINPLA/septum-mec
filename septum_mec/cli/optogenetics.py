from septum_mec.imports import *
from expipe_plugin_cinpla.scripts import utils as action_tools
from septum_mec.tools import opto as opto_tools
from expipe_plugin_cinpla.cli import utils


def attach_to_cli(cli):
    @cli.command('register', short_help='Parse optogenetics info to an action.')
    @click.argument('action-id', type=click.STRING)
    @click.option('--brain-area',
                  required=True,
                  callback=utils.optional_choice,
                  envvar=project.config.get('possible_brain_areas') or [],
                  help='The anatomical brain-area of the optogenetic stimulus.',
                  )
    @click.option('--paradigm',
                  required=True,
                  callback=utils.optional_choice,
                  envvar=project.config.get('possible_opto_paradigms') or [],
                  help='The anatomical brain-area of the optogenetic stimulus.',
                  )
    @click.option('-t', '--tag',
                  multiple=True,
                  callback=utils.optional_choice,
                  envvar=project.config.get('possible_tags') or [],
                  help='The anatomical brain-area of the optogenetic stimulus.',
                  )
    @click.option('-m', '--message',
                  multiple=True,
                  type=click.STRING,
                  help='Add message, use "text here" for sentences.',
                  )
    @click.option('--io-channel',
                  default=1,
                  type=click.INT,
                  help='TTL input channel. Default is 8 (axona tetrode 9)',
                  )
    @click.option('--no-local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite modules or not.',
                  )
    @click.option('--laser-id',
                  required=True,
                  type=click.STRING,
                  help='A unique identifier of the laser.',
                  )
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the annotation.',
                  )
    @click.option('--no-modules',
                  is_flag=True,
                  help='Do not upload any modules.',
                  )
    @click.option('--use-axona-cut',
                  is_flag=True,
                  help='Use Axona cut file for input registration.',
                  )
    @click.option('--pulse-phasedur',
                  nargs=2,
                  default=(None, None),
                  type=(click.FLOAT, click.STRING),
                  help=('Duration of laser pulse with units e.g. 10 ms.' +
                        ' Only relevant if no config file available.'),
                  )
    @click.option('--pulse-period',
                  nargs=2,
                  default=(None, None),
                  type=(click.FLOAT, click.STRING),
                  help=('Period of laser pulse with units e.g. 10 ms.' +
                        ' Only relevant if no config file available.'),
                  )
    @click.option('--no-intensity',
                  is_flag=True,
                  help='Use Axona cut file for input registration.',
                  )
    @click.option('--laser-name',
                  default='hardware_blue_laser',
                  type=click.STRING,
                  help='A name of the laser.',
                  )
    @click.option('--pulsepalfile',
                  type=click.Path(exists=True),
                  help='Find parameters from PulsePal params file.',
                  )
    def parse_optogenetics(action_id, brain_area, no_local, overwrite,
                           io_channel, tag, message, laser_id, user,
                           no_modules, use_axona_cut, pulse_phasedur,
                           pulse_period, no_intensity, paradigm, pulsepalfile,
                           laser_name):
        # TODO deafault none
        action = project.actions[action_id]
        user = user or project.config.get('username') or []
        if user is None:
            raise ValueError('Please add user name')
        action.tags.extend(list(tag) + ['opto-' + brain_area] + list(paradigm))

        exdir_path = action_tools._get_data_path(action)

        exdir_object = exdir.File(exdir_path, plugins=exdir.plugins.quantities)
        aq_sys = exdir_object['acquisition'].attrs['acquisition_system'].lower()
        if aq_sys == 'axona':
            if use_axona_cut:
                if pulse_phasedur == (None, None):
                    raise ValueError (
                        'You need to provide pulse phase duration, e.g.' +
                        '"pulse-phasedur 10 ms" to use Axona cut')
                pulse_phasedur = pq.Quantity(
                    pulse_phasedur[0], pulse_phasedur[1])
                params = opto_tools.generate_axona_opto_from_cut(
                    exdir_path, pulse_phasedur, io_channel)
            else:
                params = opto_tools.generate_axona_opto(
                    exdir_path, io_channel, no_intensity=no_intensity)
        elif aq_sys == 'openephys' or aq_sys == 'rhythm fpga':
            aq_sys = 'openephys'
            params = opto_tools.generate_openephys_opto(
                exdir_path, io_channel, pulsepalfile)
        else:
            raise ValueError('Acquisition system not recognized')
        params['paradigm'] = paradigm
        if not no_modules:
            params.update({'location': brain_area})
            action_tools.generate_templates(
                action, 'opto_' + aq_sys, overwrite=overwrite)
            opto_tools.populate_modules(
                action, params, no_intensity=no_intensity)
            action.modules[laser_name]['device_id'] = {'value': laser_id}

        for m in message:
            action.create_message(text=m, user=user, datetime=datetime.now())

    @cli.command('parse-files', short_help='Parse optogenetics files.')
    @click.argument('action-id', type=click.STRING)
    @click.option('--no-local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('--io-channel',
                  default=4,
                  type=click.INT,
                  help='TTL input channel.',
                  )
    def parse_optogenetics_files(action_id, no_local, io_channel):
        action = project.require_action(action_id)
        exdir_path = action_tools._get_data_path(action)
        exdir_object = exdir.File(exdir_path, plugins=exdir.plugins.quantities)
        if exdir_object['acquisition'].attrs['acquisition_system'] == 'Axona':
            aq_sys = 'axona'
            params = opto_tools.generate_axona_opto(exdir_path, io_channel)
        elif exdir_object['acquisition'].attrs['acquisition_system'] == 'OpenEphys':
            aq_sys = 'openephys'
            params = opto_tools.generate_openephys_opto(exdir_path, io_channel)
        else:
            raise ValueError('Acquisition system not recognized')
