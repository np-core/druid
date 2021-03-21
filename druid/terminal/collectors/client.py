import click

from .dnd_graftm import dnd_graftm


@click.group()
@click.version_option()
def collectors():
    pass


collectors.add_command(dnd_graftm)
