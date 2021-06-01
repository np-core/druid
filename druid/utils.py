import os
import h5py
import random
import datetime
import numpy as np
import matplotlib
import pandas
import subprocess
import shlex
import logging
import pyfastx

from pathlib import Path
from colorama import Fore
from matplotlib import style
from skimage.util import view_as_windows
from ont_fast5_api.fast5_file import Fast5File

Y = Fore.YELLOW
R = Fore.RED
G = Fore.GREEN
C = Fore.CYAN
M = Fore.MAGENTA
LR = Fore.LIGHTRED_EX
LC = Fore.LIGHTCYAN_EX
LY = Fore.LIGHTYELLOW_EX
RE = Fore.RESET


matplotlib.use("agg")
style.use("ggplot")

# Data IO and Transformation


class PoreLogger:

    def __init__(self, level=logging.ERROR, name: str = None):

        logging.basicConfig(
            level=level,
            format=f"[%(asctime)s] [{name}]     %(message)s",
            datefmt='%H:%M:%S',
        )

        self.logger = logging.getLogger()



def timeit(micro=False):
    def decorator(func):
        """ Timing decorator for functions and methods """

        def timed(*args, **kw):
            start_time = datetime.datetime.now()
            result = func(*args, **kw)
            time_delta = datetime.datetime.now() - start_time
            seconds = time_delta.total_seconds()
            if micro:
                seconds = int(seconds * 1000000)  # Microseconds
            # print("Runtime:", seconds, "seconds")
            # Flatten output if the output of a function is a
            # tuple with multiple items if this is the case,
            # seconds are at index -1
            return [
                num
                for item in [result, seconds]
                for num in (item if isinstance(item, tuple) else (item,))
            ]

        return timed

    return decorator

# From Mako (ONT) - GitHub site here


def med_mad(data, factor=1.4826, axis=None):

    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median.

    :param data: A :class:`ndarray` object
    :param axis: For multidimensional arrays, which axis to calculate over
    :returns: a tuple containing the median and MAD of the data
    .. note :: the default `factor` scales the MAD for asymptotically normal
        consistency as in R.

    """
    dmed = np.median(data, axis=axis)
    if axis is not None:
        dmed1 = np.expand_dims(dmed, axis)
    else:
        dmed1 = dmed

    dmad = factor * np.median(np.abs(data - dmed1), axis=axis)
    return dmed, dmad


def _scale_data(data):
    if data.ndim == 3:
        # (batches, timesteps, features)
        med, mad = med_mad(data, axis=1)
        med = med.reshape(med.shape + (1,))
        mad = mad.reshape(mad.shape + (1,))
        data = (data - med) / mad
    elif data.ndim == 1:
        med, mad = med_mad(data)
        data = (data - med) / mad
    else:
        raise AttributeError("'data' should have 3 or 1 dimensions.")
    return data


def norm(prediction):
    """ Probability normalization to 1 for predictions along multiple windows of signal """
    return [float(i) / sum(prediction) for i in prediction]


def find(key, dictionary):
    """ https://gist.github.com/douglasmiranda/5127251 """
    for k, v in dictionary.items():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result


def get_dataset_file_names(datasets):

    """ If we sample from the same (random) subset of reads as the training data, this function
    makes sure that we are not using the same files used in training for evaluation / prediction. """

    file_names = []
    for data_file in datasets:
        with h5py.File(data_file, "r") as data:
            file_names += [os.path.basename(file) for file in data["data/files"]]

    return file_names


def get_dataset_labels(dataset):

    """ If we sample from the same (random) subset of reads as the training data, this function
    makes sure that we are not using the same files used in training for evaluation / prediction. """

    with h5py.File(dataset, "r") as data:
        labels = data["data/labels"]
        return np.array(labels)


def get_dataset_dim(dataset):

    """ If we sample from the same (random) subset of reads as the training data, this function
    makes sure that we are not using the same files used in training for evaluation / prediction. """

    with h5py.File(dataset, "r") as data:
        return np.array(data["training/data"]).shape


def run_cmd(cmd, callback=None, watch=False, background=False, shell=False):

    """Runs the given command and gathers the output.

    If a callback is provided, then the output is sent to it, otherwise it
    is just returned.

    Optionally, the output of the command can be "watched" and whenever new
    output is detected, it will be sent to the given `callback`.

    Returns:
        A string containing the output of the command, or None if a `callback`
        was given.
    Raises:
        RuntimeError: When `watch` is True, but no callback is given.

    """

    if watch and not callback:
        raise RuntimeError(
            "You must provide a callback when watching a process."
        )

    output = None

    if shell:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    else:
        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE)

    if background:
        # Let task run in background and return pmid for monitoring:
        return proc.pid, proc

    if watch:
        while proc.poll() is None:
            line = proc.stdout.readline()
            if line != "":
                callback(line)

        # Sometimes the process exits before we have all of the output, so
        # we need to gather the remainder of the output.
        remainder = proc.communicate()[0]
        if remainder:
            callback(remainder)
    else:
        output = proc.communicate()[0]

    if callback and output is not None:
        return callback(output)

    return output


# TAXONOMY IDENTIFICATION

# Make sure you have downloaded ftp://ftp.ncbi.nih.gov/pub/taxonomy/taxdmp.zip
# and have its contents in ./data/.

# Note: This is just another reinvention of NCBITaxonomy of ETE Toolkit.
# http://etetoolkit.org/docs/latest/tutorial/tutorial_ncbitaxonomy.html


def _read_nodes(path: Path):
    nodes = pandas.read_csv(path / "nodes.dmp", sep="|", header=None)
    nodes = nodes.drop(nodes.columns[3:], axis=1)
    nodes.columns = ["taxid", "parentid", "rank"]
    nodes = nodes.set_index("taxid")
    nodes["rank"] = nodes["rank"].apply(lambda x: x.strip())
    return nodes


def _read_names(path: Path):
    names = pandas.read_csv(path / "names.dmp", sep="|", header=None)
    names = names.drop([names.columns[2], names.columns[4]], axis=1)
    names.columns = ["taxid", "name", "type"]
    names = names.set_index("taxid")
    names = names.applymap(lambda x: x.strip())
    names = names[names["type"] == "scientific name"]
    return names


def _read_merged(path: Path):
    merged = pandas.read_csv(path / "merged.dmp", sep="|", header=None)
    merged = merged.drop([merged.columns[2]], axis=1)
    merged.columns = ["original", "mergedto"]
    merged = merged.set_index("original")
    return merged


def get_tax(taxid, nodes, names, merged, prev_tax=None):
    if prev_tax is None:
        prev_tax = dict()  # DO NOT GIVE IT AS A DEFAULT PARAMETER
    while taxid in merged.index:  # substitute with merged taxid ITERATIVELY just in case
        taxid = merged.loc[taxid]["mergedto"]

    prev_tax[nodes.loc[taxid]["rank"]] = names.loc[taxid]["name"]

    # recursion
    if "kingdom" in nodes.loc[taxid]["rank"]:
        return prev_tax
    else:
        return get_tax(nodes.loc[taxid]["parentid"], nodes, names, merged, prev_tax)


def prep_tax(tax_path: Path):
    nodes = _read_nodes(path=tax_path)
    names = _read_names(path=tax_path)
    merged = _read_merged(path=tax_path)

    return nodes, names, merged


def parse_operon_sequences(fasta: Path) -> (list, dict, dict):

    accessions = []
    descriptions = {}
    seqs = {}
    for file in fasta.glob("*.fasta"):
        for name, seq in pyfastx.Fasta(str(file), build_index=False, full_name=True):
            acc = name.split()[0].split(":")[0].replace(">", "")
            descr = ' '.join(name.split()[1:])
            accessions.append(acc)
            descriptions[acc] = descr
            seqs[acc] = f">{acc}\n{seq}"

    return accessions, descriptions, seqs
