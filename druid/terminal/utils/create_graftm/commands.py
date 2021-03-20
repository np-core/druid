import click
import pandas
from pathlib import Path
from pyfasta import Fasta
from druid.utils import run_cmd, prep_tax, get_tax
from io import StringIO


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
    "--tax_path",
    "-t",
    type=Path,
    default=None,
    metavar="",
    help="Accession",
)
def create_graftm(fasta, name, tax_path, outdir):

    """Create a GraftM package from a set of fasta files"""

    outdir.mkdir(parents=True, exist_ok=True)

    seqs = [seq for file in fasta.glob("*.fasta") for seq in Fasta(str(file))]

    grep = "|".join([str(seq).split()[0].split(":")[0] for seq in seqs])

    output = StringIO(
        run_cmd(f"grep -E {grep} {tax_path / 'nucl_gb.accession2taxid'}").decode("utf-8")
    )

    nodes, names, merged = prep_tax(tax_path=tax_path)

    taxids = pandas.read_csv(
        output, sep='\t', header=None, names=["accession", "version", "taxid", "gi"]
    )

    for _, row in taxids.iterrows():
        taxid = row['taxid']
        print(taxid)
        print(get_tax(taxid, nodes, names, merged))

    # Write GraftM sequence file:
    with (outdir / f"{name}.fasta").open('w') as fout:
        for seq in seqs:
            fout.write(seq + '\n')




