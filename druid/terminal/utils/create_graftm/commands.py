import click
import pandas
import pyfastx
from pathlib import Path
from druid.utils import run_cmd, prep_tax, get_tax
from io import StringIO
from collections import OrderedDict

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
    "--package_name",
    "-n",
    type=str,
    default="graftm_package",
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
def create_graftm(fasta, package_name, tax_path, outdir):

    """Create a GraftM package from a set of fasta files"""

    outdir.mkdir(parents=True, exist_ok=True)

    seqs = []
    accessions = []
    for file in fasta.glob("*.fasta"):
        for name, seq in pyfastx.Fasta(str(file), build_index=False):
            acc = name.split()[0].split(":")[0].replace(">", "")
            seqs.append(f">{acc}\n{seq}")
            accessions.append(acc)

    for seq in seqs:
        print(seq)

    grep = "|".join(accessions)

    output = StringIO(
        run_cmd(f"grep -E {grep} {tax_path / 'nucl_gb.accession2taxid'}").decode("utf-8")
    )

    nodes, names, merged = prep_tax(tax_path=tax_path)

    taxids = pandas.read_csv(
        output, sep='\t', header=None, names=["accession", "version", "taxid", "gi"]
    )

    print(taxids)

    for _, row in taxids.iterrows():
        taxid = row['taxid']
        tax_hierarchy = get_tax(taxid, nodes, names, merged)
        tax_greengenes = tax_to_greengenes(tax_hierarchy)
        print(tax_greengenes)

    # Write GraftM sequence file:
    with (outdir / f"{package_name}.fasta").open('w') as fout:
        for seq in seqs:
            fout.write(seq + '\n')


def tax_to_greengenes(tax_hierarchy: dict):

    """ Convert a hierarchical taxonomy to GreenGenes format """

    convert = [
        ('superkingdom', 'k__'),
        ('phylum', 'p__'),
        ('class', 'c__'),
        ('order', 'o__'),
        ('family', 'f__'),
        ('genus', 'g__'),
        ('species', 's__')
    ]

    gg = []
    vals = []
    for i, (level, short) in enumerate(convert):
        try:
            val = tax_hierarchy[level]
        except KeyError:
            val = ''

        if level == 'species':
            try:
                val = val.split()[1]
            except IndexError:
                val = ""

            if 'sp.' in val:
                val = ""

        if 'Candidatus' in val:
            try:
                val = f"[{val.split()[1]}]"  # take first should be: Candidatus Nitrospec...
            except IndexError:
                val = ""

        # Check if there are some weird name formats and use first only
        values = val.split()
        if len(values) > 1:
            val = values[0]

        gg.append(f"{short}{val}")

    return ";".join(gg)

