import click
from pathlib import Path
from druid.pipeline import DruidPipeline


@click.command()
@click.option(
    "--directory",
    "-d",
    type=Path,
    default=Path.cwd(),
    metavar="",
    help="Path to directory of workflow output [np-core/druid --workflow graftm_search]",
)
def dnd_graftm(directory):

    """GraftM collector from the metagenome search pipeline """

    dr = DruidPipeline(directory=directory)

    data = dr.collect_graftm()
    dr.plot_graftm_counts(package_data=data, plot_name="graftm")
