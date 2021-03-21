import click
import pandas
import pyfastx
from pathlib import Path


@click.command()
@click.option(
    "--dir",
    "-d",
    type=Path,
    default=Path.cwd(),
    metavar="",
    help="Path to directory of workflow output [np-core/druid --workflow graftm_search]",
)
def dnd_graftm(dir):

    """GraftM collector from the metagenome search pipeline """

    pass
