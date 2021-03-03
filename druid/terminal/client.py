import click

from .create import create
from .train import train
from .predict import predict
from .inspect import inspect
from .evaluate import evaluate

VERSION = "0.3-alpha"


@click.group()
@click.version_option(version=VERSION)
def terminal_client():
    pass


terminal_client.add_command(create)
terminal_client.add_command(train)
terminal_client.add_command(predict)
terminal_client.add_command(inspect)
terminal_client.add_command(evaluate)