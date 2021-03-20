import click
import pandas
from pathlib import Path
from pyfasta import Fasta


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
    help="Accession to taxid resources file to add taxonomy file to GraftM from FASTA parsed from headers",
)
def create_graftm(fasta, name, acc2tax, outdir):

    """Create a GraftM package from a set of fasta files"""

    outdir.mkdir(parents=True, exist_ok=True)

    seqs = [seq for file in fasta.glob("*.fasta") for seq in Fasta(str(file))]

    for seq in seqs:
        seqid, descr = str(seq).split()
        print(seqid)
        print(descr)

    # Write GraftM sequence file:
    with (outdir / f"{name}.fasta").open('w') as fout:
        for seq in seqs:
            fout.write(seq + '\n')




