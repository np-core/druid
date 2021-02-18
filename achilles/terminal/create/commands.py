import click
import yaml

from pathlib import Path
from poremongo import PoreMongo
from achilles.dataset import AchillesDataset

@click.command()
@click.option(
    "--uri",
    "-u",
    default="local",
    help="PoreMongo connection 'local' or URI ",
    show_default=True,
    metavar="",
)
@click.option(
    "--db",
    "-db",
    default="poremongo",
    help="PoreMongo database to sample from",
    show_default=True,
    metavar="",
)
@click.option(
    "--tags",
    "-t",
    type=str,
    default=None,
    metavar="",
    help="Tags (labels) to sample from, labels separated by ':' for example, two label dataset: tag1,tag2:tag3,tag4",
)
@click.option(
    "--dataset",
    "-d",
    default="dataset.hd5",
    metavar="",
    show_default=True,
    help="Output HDF5 file containing sampled tensors and labels",
)
@click.option(
    "--max_windows",
    "-mw",
    default=100000,
    metavar="",
    show_default=True,
    help="Maximum number of sampled " "signal value windows per tag / label",
)
@click.option(
    "--max_windows_per_read",
    "-mwr",
    default=50,
    metavar="",
    show_default=True,
    help="Maximum number of windows sampled from" " read / diversity of input data",
)
@click.option(
    "--window_size",
    "-wsz",
    default=200,
    metavar="",
    show_default=True,
    help="Length of sliding window to sample from signal read",
)
@click.option(
    "--window_step",
    "-wsp",
    default=0.1,
    metavar="",
    show_default=True,
    help="Step of sliding window to sample from signal read",
)
@click.option(
    "--sample",
    "-s",
    default=10000,
    metavar="",
    show_default=True,
    help="Number of random Fast5 models to initially sample from database per tag / label",
)
@click.option(
    "--proportion",
    '-p',
    default="equal",
    metavar="",
    show_default=True,
    help="Proportion of Fast5 models to sample per tag / label",
)
@click.option(
    "--exclude",
    "-e",
    default=None,
    metavar="",
    show_default=True,
    help="Comma separated list of HDF5 datasets to exclude from sampling or glob-path to dataset directory",
)
@click.option(
    "--global_tags",
    "-g",
    default=None,
    metavar="",
    show_default=True,
    help="Global tags to apply to sample, comma-separated, e.g. to force pore version: R9.4",
)
@click.option(
    "--validation",
    "-v",
    default=0.3,
    metavar="",
    help="Proportion of data to be split into validation",
)
@click.option(
    '--pm_config',
    '-p',
    type=Path,
    metavar="",
    default=None,
    help='Path to config file for database connection'
)
@click.option(
    "--achilles_config",
    "-a",
    default=None,
    type=Path,
    metavar="",
    help="Achilles dataset configuration file",
)
def create(
    uri,
    db,
    tags,
    dataset,
    pm_config,
    achilles_config,
    max_windows,
    max_windows_per_read,
    window_size,
    window_step,
    sample,
    proportion,
    exclude,
    global_tags,
    validation
):
    """Sample and compile datasets with PoreMongo"""

    if uri == 'local':
        uri = f'mongodb://localhost:27017/{db}'

    pongo = PoreMongo(uri=uri, config=pm_config)
    pongo.connect()

    global_tags = [s.strip() for s in global_tags.split()] if global_tags else None

    if not achilles_config:

        # Manual input from command line

        ds = AchillesDataset(
            poremongo=pongo,
            max_windows=max_windows,
            max_windows_per_read=max_windows_per_read,
            window_size=window_size,
            window_step=window_step,
            window_random=True,  # sample a sequence of signal windows from random start point
            window_recover=False,  # allow incomplete windows at sequence end (slightly variable total slice samples)
            sample_reads_per_tag=sample,
            sample_proportions=proportion,
            sample_unique=False,  # can be used as safe guard
            exclude_datasets=exclude,
            validation=validation,
            chunk_size=10000,
            global_tags=global_tags
        )

        tag_labels = []
        for tag_group in tags.split(":"):
            tag_labels.append(
                [t.strip() for t in tag_group.split(",")]
            )

        ds.write(data_file=dataset, tag_labels=tag_labels)

    else:

        # input from configuration file

        config = read_yaml(yaml_file=achilles_config)

        params = config['params']
        datasets = config['datasets']

        for name, dataset_config in datasets.items():
            ds_file = f"{name}.hdf5"
            tag_labels = dataset_config['tags']
            ds = AchillesDataset(poremongo=pongo, **params)
            ds.print_write_summary()
            ds.write(tag_labels=tag_labels, data_file=ds_file)

    pongo.disconnect()


def read_yaml(yaml_file: Path):

    with yaml_file.open('r') as fstream:
        yml = yaml.safe_load(fstream)

    return yml