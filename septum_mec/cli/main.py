from expipecli.utils.plugin import IPlugin
from . import analysis
from . import optogenetics
from septum_mec.imports import *


class SeptumPlugin(IPlugin):
    def attach_to_cli(self, cli):
        @cli.group(short_help='Tools related to optogenetics.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def opto(ctx):
            pass

        @cli.group(short_help='Tools related to Open Ephys.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def openephys(ctx):
            pass

        @cli.group(short_help='Tools related.')
        @click.help_option('-h', '--help')
        @click.pass_context
        def spikeinterface(ctx):
            pass

        analysis.attach_to_cli(cli)
        OE.attach_to_cli(openephys)
        optogenetics.attach_to_cli(opto)
        remote.attach_to_cli(cli)
        sortertool.attach_to_cli(spikeinterface)
