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
@click.option(
    "--graftm_reads",
    is_flag=True,
    help="Collect GraftM results executed on reads [FASTQ]",
)
@click.option(
    "--graftm_mags",
    is_flag=True,
    help="Collect GraftM results executed on MAGs [FASTA]",
)
def dnd_graftm(directory, graftm_reads, graftm_mags):

    """GraftM collector from the metagenome search pipeline """

    dr = DruidPipeline(directory=directory)

    data = dr.collect_graftm()

    if graftm_reads:
        dr.plot_graftm_counts(package_data=data, plot_name="graftm")

    if graftm_mags:
        dr.get_graftm_mag_table(graftm_data=data)