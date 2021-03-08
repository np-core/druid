import os
import h5py
import random
import pandas
import datetime
import itertools
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from matplotlib import style

from collections import deque
from itertools import tee, islice
from pandas.errors import EmptyDataError

from ont_fast5_api.fast5_file import Fast5File
from skimage.util import view_as_windows

from colorama import Fore

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


def read_signal(
    fast5: str,
    normalize: bool = False,
    scale: bool = False,
    window_size: int = 400,
    window_step: int = 400,
    window_max: int = 10,
    window_random: bool = True,
    window_recover: bool = True,
    return_signal: bool = False,
) -> np.array:

    """ Read scaled raw signal in pA (float32) if scaling is enabled or raw (DAC, int16) values from Fast5 using ONT API

    :param fast5            str     path to .fast5 file
    :param normalize        bool    normalize signal by subtracting mean and dividing by standard deviation
    :param window_size      int     run sliding window along signal with size, pass None to return all signal values
    :param window_step      int     sliding window stride, usually 10% of window_size, but appears good on as well
                                    on non-overlapping window slides where window_step = window_size
    :param window_max       int
    :param window_random    bool
    :param window_recover   bool

    :returns tuple of window_max signal windows (np.array) and number of total signal windows before window_max
    """

    try:
        fast5 = Fast5File(fname=fast5)
    except OSError:
        # If the file can't be opened:
        return None, 0

    # Scale for array of float(pA values)
    signal = fast5.get_raw_data(scale=scale)

    if normalize:
        signal = (signal - signal.mean()) / signal.std()

    if return_signal:
        # Here, we only return the signal array (1D) and number of signals,
        # used in select function:
        return signal, len(signal)

    # Window processing part:

    signal_windows = view_as_windows(signal, window_size, window_step)

    # Select a random index to extract signal slices, subtract window_max
    # to generate a suitable index where the total number of windows is larger
    # than the requested number of windows then proceed to take a sequence
    # of windows from the random index or from start:
    nb_windows_total = len(signal_windows)
    max_index = nb_windows_total - window_max

    if max_index >= 0:
        if window_random:
            # If max_windows_per_read can be extracted...select random index:
            rand_index = random.randint(0, max_index)
            # ... and extract them:
            signal_windows = signal_windows[rand_index : rand_index + window_max, :]
        else:
            signal_windows = signal_windows[:window_max]
    else:
        # If there are fewer signal windows in the file than window_max...
        if window_recover:
            # If recovery is on, take all windows from this file,
            # as to not bias for longer reads in the sampling process for
            # generating training data:
            signal_windows = signal_windows[:]
        else:
            # Otherwise, return None and skip read, this is more
            # useful in live prediction if a read has not enough
            # (window_max) signal values written to it yet
            signal_windows = None

    return signal_windows, nb_windows_total


def transform_signal_to_tensor(array):

    """ Transform data (nb_windows,window_size) to (nb_windows, 1, window_size, 1)
    for input into Conv2D layer: (samples, height, width, channels),
    """

    # Reshape 2D array (samples, width) to 4D array (samples, 1, width, 1)
    return np.reshape(array, (array.shape[0], 1, array.shape[1], 1))


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

