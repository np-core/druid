import click

from .create_graftm import create_graftm
from .create_abricate import create_abricate


@click.group()
@click.version_option()
def utils():
    pass


utils.add_command(create_graftm)
utils.add_command(create_abricate)