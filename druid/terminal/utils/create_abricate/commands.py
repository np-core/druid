import click
from pathlib import Path
from druid.utils import parse_operon_sequences


@click.command()
@click.option(
    "--genes",
    "-g",
    type=Path,
    default=Path.cwd(),
    metavar="",
    help="Path to directory containing fasta file to process (one gene per entry)",
)
@click.option(
    "--db_name",
    "-d",
    type=str,
    default="operons",
    metavar="",
    help="Name of Abricate DB to create",
)
@click.option(
    "--outdir",
    "-o",
    type=Path,
    default=Path.cwd(),
    metavar="",
    help="Path to output directory for the GraftM package files",
)
def create_abricate(genes, db_name, outdir):

    """Create an Abricate database from a set of seqeuences"""

    outdir.mkdir(parents=True, exist_ok=True)

    gene_paths = [d for d in genes.glob("*") if d.is_dir()]

    db_seqs = []
    for gene_path in gene_paths:
        accessions, descriptions, entries = parse_operon_sequences(fasta=gene_path)
        print(entries)
        db_seqs += make_abricate_header(accessions, entries, db_name, gene_path.name)

    with (outdir / f"{db_name}.fasta").open('w') as outfile:
        for seq in db_seqs:
            outfile.write(seq + '\n')


def make_abricate_header(accessions, entries, db_name, gene_name):

    seqs = {a: e.split('\n')[1] for a, e in entries.items()}

    return [
        f">{db_name}~~~{gene_name}_{i}~~~{accession}~~~\n{seqs[accession]}"
        for i, accession in enumerate(accessions)
    ]
