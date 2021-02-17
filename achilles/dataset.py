import os
import h5py
import random
import logging
import numpy as np
import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

from tqdm import tqdm
from scipy.stats import sem
from sklearn.model_selection import train_test_split
from textwrap import dedent
from pathlib import Path
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from poremongo.poremodels import Read
from colorama import Fore

Y = Fore.YELLOW
C = Fore.CYAN
RE = Fore.RESET
G = Fore.GREEN
M = Fore.MAGENTA


style.use("ggplot")


class AchillesDataset:

    def __init__(
        self,
        dataset=None,
        poremongo=None,
        global_tags=None,
        max_windows: int = 20000,
        max_windows_per_read: int = 50,
        window_size: int = 400,
        window_step: float or int = 0.1,
        window_random: bool = True,
        window_recover: bool = True,
        sample_reads_per_tag: int = 25000,
        sample_proportions: list or str = None,
        sample_unique: bool = False,
        exclude_datasets: str = None,
        validation: float = 0.3,
        chunk_size: int = 10000,
        max_reads: int = None
    ):

        self.dataset = dataset
        self.poremongo = poremongo

        self.global_tags = global_tags
        self.max_windows = max_windows
        self.max_windows_per_read = max_windows_per_read
        self.window_size = window_size
        self.window_step = window_step
        self.window_random = window_random
        self.window_recover = window_recover
        self.sample_reads_per_tag = sample_reads_per_tag
        self.sample_proportions = sample_proportions
        self.sample_unique = sample_unique
        self.exclude_datasets = exclude_datasets
        self.validation = validation
        self.chunk_size = chunk_size
        self.max_reads = max_reads

    @staticmethod
    def get_signal_generator(
        data_file,
        data_type="training",
        batch_size=15,
        shuffle=True,
        no_labels=False,
    ):

        """Access function to generate signal window training and validation data generators
        from directories of Fast5 files, generate data in batches

        :param data_file:
        :param data_type:
        :param batch_size:
        :param shuffle:
        :param no_labels: for prediction generator in Keras
        :return:
        """

        return DataGenerator(
            data_file,
            data_type=data_type,
            batch_size=batch_size,
            shuffle=shuffle,
            no_labels=no_labels,
        )

    @staticmethod
    def get_reads_to_exclude(exclude_datasets: [Path] or None):

        if exclude_datasets is not None:
            exclude = []
            for dataset in exclude_datasets:
                print(dataset)
                with h5py.File(str(dataset), "r") as infile:
                    print(infile['data/reads'])
                    try:
                        exclude += infile["data/reads"]
                    except KeyError:
                        raise ValueError(
                            f"Could not detect path 'data/reads' in file {dataset}"
                        )
            return exclude
        else:
            return list()

    @staticmethod
    def parse(data_file):

        with h5py.File(data_file, "r") as infile:
            return infile["data/reads"], infile["data/labels"]

    def write(self, tag_labels: [[str]], data_file="data.h5"):

        if self.max_reads:
            # Optional: if specified, compute windows based on read number
            max_windows = self.max_reads * self.max_windows_per_read
        else:
            max_windows = self.max_windows

        if isinstance(self.window_step, float):
            # If provided a float, compute window step as proportion of window size
            window_step = int(self.window_size * self.window_step)
        else:
            window_step = self.window_step

        # Exclude reads from datasets in sampling strategy
        if self.exclude_datasets:
            if "*" in self.exclude_datasets:
                # Assume path: /to/dir/*
                _exclude_path = Path(self.exclude_datasets).parent
                _exclude_glob = Path(self.exclude_datasets).name
                exclude_datasets = list(_exclude_path.glob(_exclude_glob))
            else:
                # Assume string of paths: /to/file1, /to/file2
                exclude_datasets = [Path(d.strip()) for d in self.exclude_datasets.split(",")]
            ndatasets = len(exclude_datasets)
        else:
            exclude_datasets = None
            ndatasets = 0

        # Get list of Fast5 file names to exclude from sampling in PoreMongo
        exclude = self.get_reads_to_exclude(exclude_datasets)

        classes = len(tag_labels)
        with h5py.File(data_file, "w") as f:

            # Create data paths for storing all extracted data:
            data, labels, decoded, sampled_reads = self.create_data_paths(
                file=f, window_size=self.window_size, classes=classes
            )

            self.print_write_summary()

            # Each input tag corresponds to label (0, 1, 2, ...)
            for label, tags in enumerate(tag_labels):

                print(
                    dedent(f"""
                {Y}Class label: {C}{label}{Y} 
                ----------------------------------
                {G}Global tags: {C}{', '.join(self.global_tags)}{Y}
                {G}Sample tags: {C}{', '.join(tags)}{Y}
                ----------------------------------
                {Y}Exclude {C}{len(exclude)}{Y} reads from {C}{ndatasets}{Y} datasets{RE}
                """)
                )

                reads = self.poremongo.sample(
                    Read.objects,
                    tags=tags,
                    limit=self.sample_reads_per_tag,
                    proportion=self.sample_proportions,
                    unique=self.sample_unique,
                    exclude_reads=exclude,
                    include_tags=self.global_tags,
                    return_documents=True
                )

                # Randomize sampled reads, not necessary but precaution:
                random.shuffle(reads)

                total = 0
                read_ids = []

                with tqdm(total=max_windows) as pbar:
                    pbar.set_description(f"Extracting tensors for label {label}")
                    for read in reads:
                        # e.g. 100000 max_windows, 50 per read set

                        signal_windows = read.get_signal(
                            window_size=self.window_size, window_step=window_step
                        )  # full window slices

                        sampled_windows = self.sample_from_array(
                            signal_windows, sample_size=self.max_windows_per_read,
                            random_sample=self.window_random, recover=self.window_recover
                        )  # sampled contiguous window slices

                        # Proceed if the maximum number of windows per class has not been reached,
                        # and if there are windows extracted from the read:

                        if total < max_windows and sampled_windows.size > 0:
                            # If the number of extracted signal windows exceeds the difference between
                            # current total and max_windows is reached, cut off the signal window
                            # array and write it to file, to complete the loop for generating data for this label:
                            if sampled_windows.size > max_windows - total:
                                sampled_windows = sampled_windows[: max_windows - total]

                            # 4D input tensor (nb_samples, 1, signal_length, 1) for input to Residual Blocks
                            input_tensor = self.sample_to_input(sampled_windows)

                            # Write this tensor to file instead of storing in memory otherwise might raise OOM
                            self.write_chunk(data, input_tensor)

                            # Operations for update to total number of windows processed for this label,
                            # tracking files from which signal is extracted, and updating progress bar:
                            nb_windows = input_tensor.shape[0]
                            total += nb_windows
                            pbar.update(nb_windows)
                            read_ids.append(read.read_id)

                            # If the maximum number of signals for this class (label) has
                            # been reached, break the Read loop and proceed to writing
                            # label stored in memory (all at once)
                            if total >= max_windows - 1:
                                break

                self.poremongo.logger.info(f"Extracted {total} / {max_windows} windows for label {label}")

                # Writing all training labels to HDF5, as categorical (one-hot) encoding:
                encoded_labels = to_categorical(
                    np.array([label for _ in range(total)]), classes
                )
                self.write_chunk(labels, encoded_labels)

                # Decoded (label-based) encoding for dataset summary:
                decoded_labels = np.array([label for _ in range(total)])
                self.write_chunk(decoded, decoded_labels)

                # Read IDs from which signal arrays were extracted:
                read_id_array = np.array(
                    [read_id.encode("utf8") for read_id in read_ids]
                )
                # This is using internal READ ID (i.e. from Fast5) NOT UUID OR DOCUMENT ID
                # This means that if the same read was inserted multiple time (with different
                # tags, UUID or DOC ID), will not be sampled (in poremongo.sample) at all -
                # this is a strict implementation avoiding accidental sampling of the same read
                # under a different UUID or DOC ID
                self.write_chunk(sampled_reads, read_id_array)

        self.print_data_summary(data_file=data_file)

        if self.validation > 0:
            # Split dataset into training / validation data:
            self._training_validation_split(
                data_file=data_file,
                validation=self.validation,
                window_size=self.window_size,
                classes=classes,
                chunk_size=self.chunk_size
            )

    def _training_validation_split(
        self,
        data_file,
        validation: float = 0.3,
        window_size: int = 400,
        classes: int = 2,
        chunk_size: int = 10000,
    ):

        """ This function takes a complete data set generated with write_data,
        randomizes the indices and splits it into training and validation under the paths
        training/data, training/label, validation/data, validation/label.

        :param validation               proportion of data to be split into validation set
        :param window_size              window (slice) size for writing data to training file in chunks
        :param classes                  number of classes (labels)
        :param chunk_size               maximum number of windows for reading and writing in chunks
        """

        # Generate new file name for splitting data randomly into training and
        # validation data for input to AchillesModel (data_file + _training.h5)

        fname, fext = os.path.splitext(data_file)
        outfile = fname + ".training" + fext

        print("Splitting data into training and validation sets...\n")
        with h5py.File(data_file, "r") as data_file:

            # Get all indices for reading / writing in chunks:
            indices = np.arange(data_file["data/data"].shape[0])

            # Randomize the indices from data/data and split for training / validation:
            training_indices, validation_indices = train_test_split(
                indices, test_size=validation, random_state=None, shuffle=True
            )

            # Sanity checks for random and non-duplicated selection of indices:
            print("Sample of randomized training   indices:", training_indices[:5])
            print("Sample of randomized validation indices:", validation_indices[:5], "\n")

            if set(training_indices).intersection(validation_indices):
                logging.debug("Training and validation data are overlapping after splitting")
                raise ValueError("Training and validation data are overlapping after splitting")

            with h5py.File(outfile, "w") as out:
                train_x, train_y, val_x, val_y = self.create_training_validation_paths(
                    file=out, window_size=window_size, classes=classes
                )
                # Read and write the training / validation data by chunks of indices that
                # correspond to the max_windows_per_read parameter (minimum memory for processing)

                with tqdm(total=len(training_indices)) as pbar:
                    pbar.set_description("Writing training   data")
                    for i_train_chunk in self.chunk(training_indices, chunk_size):
                        self.write_chunk(
                            train_x,
                            np.take(data_file["data/data"], i_train_chunk, axis=0),
                        )
                        self.write_chunk(
                            train_y,
                            np.take(data_file["data/labels"], i_train_chunk, axis=0),
                        )
                        pbar.update(len(i_train_chunk))

                with tqdm(total=len(validation_indices)) as pbar:
                    pbar.set_description("Writing validation data")
                    for i_val_chunk in self.chunk(validation_indices, chunk_size):
                        self.write_chunk(
                            val_x, np.take(data_file["data/data"], i_val_chunk, axis=0)
                        )
                        self.write_chunk(
                            val_y,
                            np.take(data_file["data/labels"], i_val_chunk, axis=0),
                        )
                        pbar.update(len(i_val_chunk))

                self.print_data_summary(data_file=outfile)

    @staticmethod
    def plot_signal_distribution(
        data_file,
        random_windows=True,
        nb_windows=10000,
        data_path="data",
        limit=(0, 300),
        histogram=False,
        bins=None,
        stats=True,
    ):

        """ Plotting function to generate signal value histograms for each category, sampled randomly
        this operates on the standard data path, but can be changed to training / validation data paths in HDF5 """

        with h5py.File(data_file, "r") as data_file:
            # Get all indices from data path in HDF5
            indices = np.arange(data_file[data_path + "/data"].shape[0])
            # Randomize indices:
            if random_windows:
                np.random.shuffle(indices)

            # Select chunk size indices...
            indices = indices[:nb_windows]
            # ... and extract into memory:
            data_chunk = np.take(data_file[data_path + "/data"], indices, axis=0)
            label_chunk = np.take(data_file[data_path + "/labels"], indices, axis=0)
            # Transform one-hot encoded labels and get unique labels:
            all_labels = np.argmax(label_chunk, axis=1)
            # Labels should be integers starting at 0, so sort them for plot legend:
            unique_labels = sorted(np.unique(all_labels))

            # For each label, extract corresponding data chunk and flatten into simple array,
            # then plot as histogram or kernel density estimate with Seaborn (easier to see):
            for label in unique_labels:
                i = np.where(all_labels == label)[0]
                # Extract data label-wise from chunk...
                data = np.take(data_chunk, i, axis=0)
                # ... then flatten into one-dimensional array:
                data = data.flatten()

                if limit:
                    # Print percentage of reads exceeding limits:
                    below_limit = round(len(data[data < limit[0]]), 6)
                    above_limit = round(len(data[data > limit[1]]), 6)
                    print(
                        "Limit warning: found {}% ({}) signal values < {} and "
                        "{}% ({}) signal values > {} for label {}".format(
                            round((below_limit / len(data)) * 100, 6),
                            below_limit,
                            limit[0],
                            round((above_limit / len(data)) * 100, 6),
                            above_limit,
                            limit[1],
                            label,
                        )
                    )
                    # Subset the data by limits:
                    data = data[(data > limit[0]) & (data < limit[1])]

                if stats:
                    mean = data.mean()
                    standard_error = sem(data)
                    print(
                        "Label {}: {} +- {}".format(
                            label, round(mean, 6), round(standard_error, 4)
                        )
                    )

                # Plot signal values:
                if histogram:
                    sns.distplot(data, kde=False, bins=bins)
                else:
                    sns.kdeplot(data, shade=True)

            plt.legend(unique_labels, title="Label")

    @staticmethod
    def chunk(seq, size):

        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    @staticmethod
    def create_data_paths(file, window_size=400, classes=2):

        # HDF5 file dataset creation:
        data = file.create_dataset(
            "data/data",
            shape=(0, 1, window_size, 1),
            maxshape=(None, 1, window_size, 1),
        )
        labels = file.create_dataset(
            "data/labels", shape=(0, classes), maxshape=(None, classes)
        )

        # For data set summary only:
        dt = h5py.special_dtype(vlen=str)

        decoded = file.create_dataset("data/decoded", shape=(0,), maxshape=(None,))
        extracted = file.create_dataset(
            "data/reads", shape=(0,), maxshape=(None,), dtype=dt
        )

        return data, labels, decoded, extracted

    @staticmethod
    def create_training_validation_paths(file, window_size=400, classes=2):

        data_paths = {"training": [], "validation": []}

        for data_type in data_paths.keys():
            # HDF5 file dataset creation:
            data = file.create_dataset(
                data_type + "/data",
                shape=(0, 1, window_size, 1),
                maxshape=(None, 1, window_size, 1),
            )

            labels = file.create_dataset(
                data_type + "/labels", shape=(0, classes), maxshape=(None, classes)
            )

            data_paths[data_type] += [data, labels]

        return (
            data_paths["training"][0],
            data_paths["training"][1],
            data_paths["validation"][0],
            data_paths["validation"][1],
        )

    @staticmethod
    def print_data_summary(data_file):

        print(
            dedent(
                f"""
        {Y}Dataset:              :  {C}{data_file}{RE}
        {Y}Encoded label vector  :  {C}/data/labels{RE}
        {Y}Decoded label vector  :  {C}/data/decoded{RE}
        {Y}Sampled reads         :  {C}/data/reads{RE}
        """
            )
        )

        with h5py.File(data_file, "r") as f:

            if "data/data" in f.keys():
                msg = dedent(
                    f"""

                    {G}Data:       {Y}{f["data/data"].shape}{RE}
                    {G}Labels:     {Y}{f["data/labels"].shape}{RE}
                    {G}Reads:      {Y}{f["data/reads"].shape}{RE}
                    """
                )

            elif "training/data" in f.keys() and "validation/data" in f.keys():
                msg = dedent(
                    f"""

                    Training Dimensions:

                    {G}Data:       {Y}{f["training/data"].shape}{RE}
                    {G}Labels:     {Y}{f["training/labels"].shape}{RE}

                    Validation Dimensions:

                    {G}Data:       {Y}{f["validation/data"].shape}{RE}
                    {G}Labels:     {Y}{f["validation/labels"].shape}{RE}

                    """
                )
            else:
                logging.debug(
                    "Could not access either data/data or training/data + "
                    "validation/data in HDF5."
                )
                raise KeyError(
                    "Could not access either data/data or training/data + "
                    "validation/data in HDF5."
                )

            print(msg)

    @staticmethod
    def write_chunk(dataset, data):

        dataset.resize(dataset.shape[0] + data.shape[0], axis=0)
        dataset[-data.shape[0] :] = data

        return dataset

    @staticmethod
    def sample_from_array(
        array: np.array,
        sample_size: int,
        random_sample: bool = True,
        recover: bool = True,
    ) -> np.array:

        """Return a contiguous sample from an array of signal windows

        :param array
        :param sample_size
        :param random_sample
        :param recover

        """

        num_windows = array.shape[0]

        if num_windows < sample_size and recover:
            return array

        if num_windows < sample_size and not recover:
            raise ValueError(
                f"Could not recover read array: number of read windows "
                f"({num_windows}) < sample size ({sample_size})"
            )

        if num_windows == sample_size:
            return array
        else:
            if random_sample:
                idx_max = num_windows - sample_size
                rand_idx = random.randint(0, idx_max)

                return array[rand_idx: rand_idx + sample_size]
            else:
                return array[:sample_size]

    @staticmethod
    def sample_to_input(array: np.array) -> np.array:

        """ Transform input array of (number_windows, window_size) to (number_windows, 1, window_size, 1)
        for input into convolutional layers: (samples, height, width, channels) in AchillesModel
        """

        if array.ndim != 2:
            raise ValueError(
                f"Array of shape {array.shape} must conform to shape: (number_windows, window_size)"
            )

        return np.reshape(array, (array.shape[0], 1, array.shape[1], 1))

    def print_write_summary(self):

        print(
            dedent(
                f"""
        Generating data set for input to AchillesModel.
        =========================================================

        Sampling from PoreMongo (shuffled after sampling):

            - {G}sample_reads_per_tag{RE}     {Y}{self.sample_reads_per_tag}{RE} 
            - {G}sample_proportions{RE}       {Y}{self.sample_proportions}{RE} 
            - {G}sample_unique{RE}            {Y}{self.sample_unique}{RE} 

        Generating tensors of shape {Y}({self.max_windows_per_read}, 1, {self.window_size}, 1){RE} per read.
        
        For each label, sampling consecutive signal slices with the following parameters:

            - {G}max_windows{RE}              {Y}{self.max_windows}{RE}
            - {G}window_size{RE}              {Y}{self.window_size}{RE}
            - {G}window_step{RE}              {Y}{self.window_step}{RE}
            - {G}window_random{RE}            {Y}{self.window_random}{RE}           

        =========================================================
        """
            )
        )


class DataGenerator(Sequence):
    def __init__(
        self,
        data_file,
        data_type="training",
        batch_size=15,
        shuffle=True,
        no_labels=False,
    ):
        self.data_file = data_file
        self.data_type = data_type

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.no_labels = no_labels

        self.indices = []

        self.data_shape, self.label_shape = self._get_data_shapes()

        self.on_epoch_end()

    def _get_data_shapes(self):
        with h5py.File(self.data_file, "r") as f:
            return (
                f[self.data_type + "/data"].shape,
                f[self.data_type + "/labels"].shape,
            )

    def __len__(self):
        """ Number of batches per epoch """

        return int(np.floor(self.data_shape[0]) / self.batch_size)

    def __getitem__(self, index):
        """ Generate one batch of data """

        # Generate indexes of the batch
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        data, labels = self.__data_generation(indices)

        # Testing print statements:

        # print("Training data batch:", data.shape)
        # print("Training label batch:", labels.shape)
        # print("Generated data for indices:", indices)

        if self.no_labels:
            return data
        else:
            return data, labels

    def on_epoch_end(self):
        """ Updates indexes after each epoch """

        self.indices = np.arange(self.data_shape[0])

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, indices):
        """ Generates data containing batch_size samples """

        with h5py.File(self.data_file, "r") as f:
            file_data = f[self.data_type + "/data"]
            data = np.take(file_data, indices, axis=0)

            file_labels = f[self.data_type + "/labels"]
            labels = np.take(file_labels, indices, axis=0)

            return data, labels
