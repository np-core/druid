import click
from pathlib import Path
from druid.masif import ProteinModel


@click.command()
@click.option(
    "--pdb_id",
    "-p",
    type=str,
    default=None,
    required=True,
    help="PDB protein identifier to pull structure [required]",
)
@click.option(
    "--chains",
    "-c",
    type=str,
    default=None,
    required=False,
    help="Chains to get from structure, e.g. AB [none]",
)
@click.option(
    "--outdir",
    "-o",
    type=Path,
    default=Path('pdb_analysis'),
    required=False,
    help="Output directory for analysis and results [pdb_analysis]",
)
def run_masif(pdb_id, chains, outdir):

    """Run the MaSIF feature preparation and surface prediction pipeline """

    pm = ProteinModel(pdb_id=pdb_id, chains=chains, outdir=outdir)

    pm.prepare_feature_data()