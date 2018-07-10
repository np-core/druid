import os
import json
import pickle

from achilles.utils import plot_pevaluate_runner
from achilles.analysis import evaluate_predictions

""" Runners to compare parameters in prediction evaluation and other tasks of Achilles """

# Directories must be in order of labels in training: e.g. 0, 1
# Might want to disable TF CPU warning on UNIX: export TF_CPP_MIN_LOG_LEVEL=2

pevaluate_config = {

    "dirs": ["bp", "human"],

    "models": {
        "min5.h5": "dac"
    },

    "windows": {
        "min": 5,
        "max": 20,
        "step": 5
    },

    "sample_locations": ["start", "random"],

    "batch_size_max": 1000,

    "window_size": 400,
    "window_step": 400,
    "stdout": False

}

sampler_config = {

    "exclude": ["FAB3218"],

    "training": {
        "n": 10000,
        "pathogen": {
            "zibra_1": ["zibra_flowcell_1"]
        },
        "host": {
            "minimal": ["human_chr2", "human_chr4"],
            "diverse": ["human_chr2", "human_chr4", "human_chr8", "human_chr14", "human_chr20", "human_chrX"]
        },
    },

    "evaluation": {
        "n": 5000,
        "pathogen": {
            "zibra_2": ["zibra_flowcell_2"],
            "zibra_3": ["zibra_flowcell_3"],
            "zibra_4": ["zibra_flowcell_4"],
            "zibra_5": ["zibra_flowcell_5"]
        },
        "host": {
            "minimal_same": ["human_chr2", "human_chr4"],  # Independent samples for evaluation!
            "diverse_same": ["human_chr2", "human_chr4", "human_chr8", "human_chr14", "human_chr20", "human_chrX"],
            "minimal_general": ["human_chr3", "human_chr5"],
            "diverse_general": ["human_chr3", "human_chr5", "human_chr9", "human_chr15", "human_chr21", "human_chrY"]
        }
    },


    "simulation":  {
        "host": ["patient_sample_1"],
        "pathogen": ["zibra_flowcell_2", "zibra_flowcell_3", "zibra_flowcell_4", "zibra_flowcell_5"]
    }
}


def sampling_runner(data_dir=".", config="zika_sampler.json", outdir="zika"):

    """ This runner operates on a config file to generate combinations of training and
    independent prediction evaluation data / simulation of real human sample data for evaluation.
    Input data dir is searched recursively for input Fast5 directories. """

    pass


def process_sampling_config(config, data_dir):

    """ Generate a dictionary for sampling with task: select """

    if not isinstance(config, dict):
        with open(config, "r") as config_file:
            config = json.load(config_file)

    # Global sampling parameters:

    parameters = {
        "exclude": config["exclude"]
    }







def plot_runner_results(pickle_file, class_labels, runner="pevaluate"):

    with open(pickle_file, "rb") as pickled_results:
        results = pickle.load(pickled_results)

    if runner == "pevaluate":
        plot_pevaluate_runner(results, class_labels)


def pevaluate_runner(config="pevaluate.json", outdir="run", class_labels=None):

    """ Runner for prediction evaluation, compares and plots confusion matrices
    across the following combination of parameters:

        * model architecture + signal type
        * number of signal windows
        * sample location

    The configuration for the runner is described in a configuration JSON
    with the following structure:


    """

    if not isinstance(config, dict):
        with open(config, "r") as config_file:
            config = json.load(config_file)

    if class_labels is None:
        labels = [i for i, _ in enumerate(config["dirs"])]

    if config["windows"]["min"] < 1:
        raise ValueError("Minimum number of windows is 1.")

    if config["windows"]["min"] == 1:
        # Start from 0 based indexing of numbers to get the right step:
        config["windows"]["min"] = 0

    windows = list(range(config["windows"]["min"], config["windows"]["max"]+1, config["windows"]["step"]))

    if windows[0] == 0:
        # If we start from user-defined 1 or 0 (to get right step)
        # replace first entry (== 0) with 1
        windows[0] = 1

    os.makedirs(outdir)

    # Initial message:
    nb_models = len(config["models"].keys())
    nb_locations = len(config["sample_locations"])
    nb_window_numbers = len(windows)

    print("Running a total of {} prediction combinations of {} models, {} sampling locations "
          "and {} different window numbers."
          .format(nb_models*nb_locations*nb_window_numbers, nb_models, nb_locations, nb_window_numbers))

    confusions = {}
    for model, signal_type in config["models"].items():

        # Data type pA or DAC
        if signal_type == "pa":
            scale = True
        elif signal_type == "dac":
            scale = False
        else:
            raise ValueError("Model signal type must be one of: pa, dac.")

        for sample_location in config["sample_locations"]:

            # Sample location for slices: start of read or random,
            # might have influence on amplicons with barcodes, possibly
            if sample_location == "random":
                random_sample = True
            elif sample_location == "start":
                random_sample = False
            else:
                raise ValueError("Sample location must be one of: random, start")

            # Number of windows in slice, with windows size and step from config
            for number_windows in windows:
                print("Running predictions over {} windows ({}, {}, {})"
                      .format(number_windows, model, signal_type, sample_location))

                # Using semicolon as delimiter, so it does not interfere with model names:
                prefix = "{}:{}:{}:{}".format(model, signal_type, sample_location, number_windows)

                # Set number of batches (batch size) for the number of windows:
                batches = config["batch_size_max"]//number_windows

                print("Number of files in batch:", batches)
                cm, mu = evaluate_predictions(dirs=config["dirs"], model=model, prefix=os.path.join(outdir, prefix),
                                              scale=scale, window_random=random_sample, window_max=number_windows,
                                              window_size=config["window_size"], window_step=config["window_step"],
                                              batches=batches, stdout=config["stdout"], class_labels=class_labels)

                print("Average time of prediction per batch: {} microseconds.".format(mu))

                confusions[prefix] = {"confusion_matrix": cm, "average_prediction_time": mu, "batches": batches}

    with open(os.path.join(outdir, "results.pkl"), "wb") as result_pickle:
        pickle.dump(confusions, result_pickle)
