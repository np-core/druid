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
    help="Accession to taxid resources file to add taxonomy file to GraftM from FASTA parsed from headers",
)
def create_graftm(fasta, name, acc2tax, outdir):

    """Create a GraftM package from a set of fasta files"""

    outdir.mkdir(parents=True, exist_ok=True)

    seqs = []
    for file in fasta.glob("*.fasta"):
        for seq in sequences.file_reader(str(file)):
            print(seq)
            seqs.append(seq)

    for seq in seqs:
        print(seq.id)

    # Write GraftM sequence file:
    with (outdir / f"{name}.fasta").open('w') as fout:
        for seq in seqs:
            print(seq.id)
            fout.write(str(seq) + '\n')




