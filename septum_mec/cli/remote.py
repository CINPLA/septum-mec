# from expipe_plugin_cinpla.tools.ssh import get_login, login, ssh_execute, untar
from expipe_plugin_cinpla.imports import *
from expipe_plugin_cinpla.tools import action as action_tools


def attach_to_cli(cli):
    @cli.command('trancess', short_help='Transfer a dataset and process it')
    @click.argument('action-id', type=click.STRING)
    @click.option('--no-trash',
                  is_flag=True,
                  help='Do not send local data to trash after transfer.',
                  )
    @click.option('--overwrite',
                  is_flag=True,
                  help='Overwrite data or not.',
                  )
    @click.option('--port',
                  default=22,
                  type=click.INT,
                  help='SSH port. Default is 22',
                  )
    @click.option('--hostname',
                  type=click.STRING,
                  help='SSH hostname.',
                  )
    @click.option('--username',
                  type=click.STRING,
                  help='SSH username.',
                  )
    @click.option('--server',
                  default='processing',
                  type=click.STRING,
                  help='Name of server as named in config.yaml. Default is "nird"',
                  )
    @click.option('-y', '--yes',
                  is_flag=True,
                  default=False,
                  help='disables yes/no prompts for automated removal of files after transfer. Default is False')
    def transcess(action_id, overwrite, no_trash, port, username,
                 hostname, server, yes):
        assert server in expipe.config.settings
        server_dict = expipe.config.settings.get(server)
        project = expipe_server.require_project(PAR.PROJECT_ID)
        action = project.actions[action_id]
        fr = action.require_filerecord()

        host, user, pas, port = get_login(
            hostname=hostname, username=username, port=port, server=server_dict)
        ssh, scp_client, sftp_client, pbar = login(
            hostname=host, username=user, password=pas, port=port)
        serverpath = expipe.config.settings[server]['data_path']
        server_data = os.path.dirname(os.path.join(serverpath, fr.exdir_path))
        server_data = server_data.replace('\\', '/')
        ########################## SEND  #######################################
        local_data = os.path.dirname(action_tools._get_local_path(fr, assert_exists=True))
        print('Initializing transfer of "' + local_data + '" to "' +
              server_data + '"')
        try: # make directory for untaring
            sftp_client.mkdir(server_data)
        except IOError:
            pass
        print('Packing tar archive')
        shutil.make_archive(local_data, 'tar', local_data)
        scp_client.put(
            local_data + '.tar', server_data + '.tar', recursive=False)
        try:
            pbar[0].close()
        except Exception:
            pass
        print('Unpacking tar archive')
        cmd = "tar -C " + server_data + " -xf " + server_data + '.tar'
        if not overwrite:
            cmd += " -k --skip-old-files"
        else:
            cmd += " -k --overwrite"
        ssh_execute(ssh, cmd)
        print('Deleting tar archives')
        sftp_client.remove(server_data + '.tar')
        os.remove(local_data + '.tar')
        ###################### PROCESS #######################################
        print('Processing on server')
        ssh_execute(ssh, "expipe openephys process {}".format(action_id), get_pty=True, timeout=None)
        ####################### RETURN PROCESSED DATA #######################
        local_data = os.path.dirname(action_tools._get_local_path(fr, make=True))
        print('Initializing transfer of "' + server_data + '" to "' +
                      local_data + '"')
        print('Packing tar archive')
        exclude_statement = ""
        ssh_execute(ssh, "tar --exclude=acquisition -cf " +
                    server_data + '.tar ' + server_data)
        scp_client.get(server_data + '.tar', local_data + '.tar',
                       recursive=False)
        try:
            pbar[0].close()
        except Exception:
            pass
        print('Unpacking tar archive')
        untar(local_data + '.tar', server_data) # TODO merge with existing
        print('Deleting tar archives')
        os.remove(local_data + '.tar')
        sftp_client.remove(server_data + '.tar')
        ##################### CLOSE UP #############################
        ssh.close()
        sftp_client.close()
        scp_client.close()
