import click

from .plots import plots
from .utils import utils
from .protein import protein
from .collectors import collectors

@click.group()
@click.version_option()
def terminal_client():
    pass


terminal_client.add_command(protein)
terminal_client.add_command(collectors)
terminal_client.add_command(plots)
terminal_client.add_command(utils)