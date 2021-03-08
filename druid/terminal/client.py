import click

from .utils import utils
from .create import create
from .train import train
from .evaluate import evaluate

@click.group()
@click.version_option()
def terminal_client():
    pass


terminal_client.add_command(utils)
terminal_client.add_command(create)
terminal_client.add_command(train)
terminal_client.add_command(evaluate)