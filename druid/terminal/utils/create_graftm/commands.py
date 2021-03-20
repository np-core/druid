import click
import pandas
from pathlib import Path
from pyfasta import Fasta
from druid.utils import run_cmd

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
    "--nucl_gb",
    "-n",
    type=Path,
    default=None,
    metavar="",
    help="Accession",
)
def create_graftm(fasta, name, nucl_gb, outdir):

    """Create a GraftM package from a set of fasta files"""

    outdir.mkdir(parents=True, exist_ok=True)

    seqs = [seq for file in fasta.glob("*.fasta") for seq in Fasta(str(file))]

    grep = "|".join([str(seq).split()[0].split(":")[0] for seq in seqs])

    print(grep)
    print(nucl_gb)

    output = run_cmd(f"grep -E {grep} {nucl_gb}")

    print(output)

    # Write GraftM sequence file:
    with (outdir / f"{name}.fasta").open('w') as fout:
        for seq in seqs:
            fout.write(seq + '\n')




