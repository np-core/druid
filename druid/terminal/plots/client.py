import click

from .plot_graftm import plot_graftm


@click.group()
@click.version_option()
def plots():
    pass


plots.add_command(plot_graftm)
