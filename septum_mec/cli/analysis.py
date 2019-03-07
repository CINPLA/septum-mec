from septum_mec.imports import *
from expipe_plugin_cinpla.scripts import utils as action_tools
from septum_mec.analysis.analyser import Analyser
from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.cli import utils
ANALYSIS_PARAMS = {}
def deep_update(d, other):
    for k, v in other.items():
        d_v = d.get(k)
        if (isinstance(v, collections.Mapping) and
            isinstance(d_v, collections.Mapping)):
            deep_update(d_v, v)
        else:
            d[k] = copy.deepcopy(v)

def attach_to_cli(cli):
    @cli.command('generate-notebook',
                 short_help=("Make a notebook from template and put it in" +
                             " the analysis directory of respective action."))
    @click.argument('action-id', type=click.STRING)
    @click.option('--no_local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('--channel-group',
                  type=click.INT,
                  help='Which channel-group to analyse.',
                  )
    @click.option('--run',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    def generate_notebook(action_id, channel_group, no_local, run):
        from septum_mec.analysis.utils import create_notebook
        action = project.require_action(action_id)
        exdir_path = action_tools._get_data_path(action)
        fname = create_notebook(exdir_path)
        if run:
            subprocess.run(['jupyter', 'notebook', fname])

    @cli.command('analyse', short_help='Analyse a dataset.')
    @click.argument('action-id', type=click.STRING)
    @click.option('--channel-group',
                  multiple=True,
                  type=click.INT,
                  help='Which channel-group to analyse.',
                  )
    @click.option('--no-local',
                  is_flag=True,
                  help='Store temporary on local drive.',
                  )
    @click.option('-a', '--analysis',
                  multiple=True,
                  type=click.Choice(['spike-stat', 'spatial', 'all',
                                     'psd', 'spike-lfp', 'tfr', 'stim-stat',
                                     'occupancy']),
                  help='Analyse data.',
                  )
    @click.option('-t', '--tag',
                  multiple=True,
                  type=click.STRING,
                  callback=utils.optional_choice,
                  envvar=project.config.get('possible_tags') or [],
                  help='Add tags to action.',
                  )
    @click.option('-m', '--message',
                  multiple=True,
                  type=click.STRING,
                  help='Add message, use "text here" for sentences.',
                  )
    @click.option('-u', '--user',
                  type=click.STRING,
                  help='The experimenter performing the adjustment.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite.',
                  )
    @click.option('--hard',
                  is_flag=True,
                  help='Overwrite hard - delete analysis action.',
                  )
    @click.option('--skip',
                  is_flag=True,
                  help='Skip previously generated files.',
                  )
    def analysis(**kwargs):
        if len(kwargs['channel_group']) == 0: kwargs['channel_group'] = None
        action_id = kwargs['action_id'] + '-analysis'
        action = project.require_action(action_id)
        if kwargs['overwrite'] and kwargs['hard']:
            try:
                project.delete_action(action_id)
            except NameError as e:
                print(str(e))
        rec_action = project.require_action(kwargs['action_id'])
        action.type = 'Action-analysis'
        user = kwargs['user'] or project.config.get('username') or []
        user = user or []
        if len(user) == 0:
            raise ValueError('Please add user name')

        users = list(set(rec_action.users))
        if user not in users:
            users.append(user)
        action.users = users
        action.tags.extend(list(kwargs['tag']) + list(rec_action.tags))
        action.location = rec_action.location or ''
        action.datetime = rec_action.datetime or ''
        entities = rec_action.entities or []
        action.entities.extend(list(entities))
        for m in kwargs['message']:
            action.create_message(text=m, user=user, datetime=datetime.now())
        exdir_path = action_tools._get_data_path(project.actions[kwargs['action_id']])
        an = Analyser(exdir_path, params=ANALYSIS_PARAMS,
                      unit_info=None,#TDODO PAR.UNIT_INFO,
                      channel_group=kwargs['channel_group'],
                      no_local=kwargs['no_local'],
                      overwrite=kwargs['overwrite'],
                      skip=kwargs['skip'])
        if any(arg in kwargs['analysis'] for arg in ['stim-stat', 'all']):
            print('Analysing stimulation statistics.')
            an.stimulation_statistics()
        # if any(arg in kwargs['analysis'] for arg in ['occupancy', 'all']):
        #     print('Analysing occupancy.')
        #     an.occupancy()
        if any(arg in kwargs['analysis'] for arg in ['spatial', 'all']):
            print('Analysing spatial statistics.')
            an.spatial_overview()
        if any(arg in kwargs['analysis'] for arg in ['spike-stat', 'all']):
            print('Analysing spike statistics.')
            an.spike_statistics()
        if any(arg in kwargs['analysis'] for arg in ['psd', 'all']):
            print('Analysing stimulation statistics.')
            an.psd()
        if any(arg in kwargs['analysis'] for arg in ['spike-lfp', 'all']):
            print('Analysing spike LFP relations.')
            an.spike_lfp_coherence()
        if any(arg in kwargs['analysis'] for arg in ['tfr']):
            print('Analysing TFR.')
            an.tfr()
        # EXPIPE
        if (kwargs['overwrite'] or kwargs['skip']):
            action.modules['software_analysis_parameters'] = ANALYSIS_PARAMS
        for key, val in an.analysis_output.items():
            try:
                mod = action.modules[key].contents
            except KeyError:
                mod = {}
            deep_update(mod, val)
            action.modules[key] = mod

    # @cli.command('group-analyse',
    #              short_help=('Search and generate an analysis-action that' +
    #                          ' represents and points to multiple dataset.'))
    # @click.argument('action-id', type=click.STRING)
    # @click.option('-u', '--user',
    #               type=click.STRING,
    #               help='The experimenter performing the analysis.',
    #               )
    # @click.option('-t', '--tags',
    #               multiple=True,
    #               type=click.STRING,
    #               help='Tags to sort the analysis.',
    #               )
    # @click.option('-a', '--actions',
    #               multiple=True,
    #               type=click.STRING,
    #               help='Actions to include in the analysis.',
    #               )
    # @click.option('-i', '--entities',
    #               multiple=True,
    #               type=click.STRING,
    #               help='entities to sort the analysis.',
    #               )
    # @click.option('-l', '--locations',
    #               multiple=True,
    #               type=click.STRING,
    #               help='entities to sort the analysis.',
    #               )
    # @click.option('--overwrite',
    #               is_flag=True,
    #               help='Overwrite.',
    #               )
    # def group_analysis(action_id, user, tags, overwrite, entities,
    #                    locations, actions):
    #     project = expipe.get_project(PAR.PROJECT_ROOT)
    #     analysis_action = project.require_action(action_id)
    #
    #     analysis_action.type = 'Group-analysis'
    #     user = user or PAR.USERNAME
    #     if user is None:
    #         raise ValueError('Please add user name')
    #     if len(user) == 0:
    #         raise ValueError('Please add user name')
    #     analysis_action.users.append(user)
    #     analysis_action.tags = list(tag)
    #     # TODO this is slow, can omit loading all the modules for each action
    #     for action in project.actions:
    #         if action.type != 'Action-analysis':
    #             continue
    #         if len(actions) > 0:
    #             if action.id not in actions:
    #                 continue
    #         if len(action.tags) == 0:
    #             raise ValueError('No tags in "' + action.id + '"')
    #         if not any(t in tags for t in action.tags):
    #             continue
    #         if len(entities) > 0:
    #             if not any(s in entities for s in action.entities):
    #                 continue
    #         if len(locations) > 0:
    #             if action.location not in locations:
    #                 continue
    #         name = action.id.rstrip('-analysis')
    #         analysis_action.entities.extend(list(action.entities))
    #         contents = {}
    #         for key, val in action.modules.items():
    #             if 'channel_group' in key:
    #                 contents[key] = val
    #         analysis_action.require_module(name=name, contents=contents,
    #                                        overwrite=overwrite)
    #
    # @cli.command('spikesort', short_help='Spikesort with klustakwik.')
    # @click.argument('action-id', type=click.STRING)
    # @click.option('--no-local',
    #               is_flag=True,
    #               help='Store temporary on local drive.',
    #               )
    # def spikesort(action_id, no_local):
    #     # anoying!!!!
    #     import logging
    #     from phycontrib.neo.model import NeoModel
    #     logger = logging.getLogger('phy')
    #     logger.setLevel(logging.DEBUG)
    #     ch = logging.StreamHandler(sys.stdout)
    #     ch.setLevel(logging.DEBUG)
    #     logger.addHandler(ch)
    #
    #     project = expipe.get_project(PAR.PROJECT_ROOT)
    #     action = project.require_action(action_id)
    #     exdir_path = action_tools._get_data_path(action)
    #     print('Spikesorting ', exdir_path)
    #     model = NeoModel(exdir_path)
    #     channel_groups = model.channel_groups
    #     for channel_group in channel_groups:
    #         if not channel_group == model.channel_group:
    #             model.load_data(channel_group)
    #         print('Sorting channel group {}'.format(channel_group))
    #         clusters = model.cluster(np.arange(model.n_spikes), model.channel_ids)
    #         model.save(spike_clusters=clusters)
