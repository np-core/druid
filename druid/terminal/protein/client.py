import click

from .run_masif import run_masif


@click.group()
@click.version_option()
def protein():
    """ Everything proteinaceous """
    pass


protein.add_command(run_masif)
