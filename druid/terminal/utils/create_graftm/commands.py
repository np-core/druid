import click
import pandas
from pathlib import Path
from pyfastaq import sequences


@click.command()
@click.option(
    "--fasta",
    "-f",
    type=Path,
    default=Path.cwd(),
    metavar="",
    help="Path to directory containing fasta file to process (one gene per entry)",
)
@click.option(
    "--name",
    "-n",
    type=str,
    default="grftm_pckg",
    metavar="",
    help="Name of GraftM package to create",
)
@click.option(
    "--outdir",
    "-o",
    type=Path,
    default=Path.cwd(),
    metavar="",
    help="Path to output directory for the GraftM package files",
)
@click.option(
    "--acc2tax",
    "-a",
    type=Path,
    default=None,
    metavar="",
    help="Accession to taxid reosurce file to add taxonomy file to GraftM from FASTA parsed from headers",
)
def create_graftm(fasta, name, acc2tax, outdir):

    """Creat a GraftM package from a set of fasta files"""

    outdir.mkdir(parents=True, exist_ok=True)
    seqs = [
        seq for file in fasta.glob("*.fasta")
        for seq in sequences.file_reader(str(file))
    ]

    # Write GraftM sequence file:
    writer = sequences.file_writer(str(outdir / f"{name}.fasta"))
    for seq in seqs:
        writer.write(seq)

    for seq in seqs:
        print(seq)



