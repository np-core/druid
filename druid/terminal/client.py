import click

from .utils import utils

@click.group()
@click.version_option()
def terminal_client():
    pass


terminal_client.add_command(utils)