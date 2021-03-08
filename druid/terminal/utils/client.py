import click

from .create_graftm import create_graftm


@click.group()
@click.version_option()
def utils():
    pass


utils.add_command(create_graftm)
