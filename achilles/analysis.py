import os
import sys
import numpy
import pandas

from sklearn.metrics import confusion_matrix

from achilles.model import Achilles
from achilles.dataset import Dataset

from achilles.utils import read_signal, transform_signal_to_tensor, plot_confusion_matrix, sliding_window, norm
from achilles.select import get_recursive_files, check_include_exclude


def evaluate(data_files: list, models: list, batch_size: int=100, workers: int=2, data_path: str="data",
             write: str="") -> pandas.DataFrame:

    results = {}

    for model in models:
        results[model] = {}

        achilles = Achilles()
        achilles.load_model(model_file=model)

        for file in data_files:
            ds = Dataset(data_file=file)
            eval_gen = ds.get_signal_generator(data_type=data_path, batch_size=batch_size, shuffle=True)

            loss, acc, seconds = achilles.evaluate(eval_generator=eval_gen, workers=workers)

            results[model][file] = {
                "seconds": seconds,
                "accuracy": round(acc, 4),
                "loss": round(loss, 4)
            }

            print("""
            Model:      {}
            Data:       {}
            Loss:       {}  
            Accuracy:   {} %
            Time:       {} seconds
            """.format(model, file, loss, acc*100, seconds))

    # Multi-index dataframe Model / File
    df = pandas.DataFrame.from_dict({(i, j): results[i][j] for i in results.keys() for j in results[i].keys()},
                                    orient="index").reset_index()

    # Inplace is necessary for underlying objects:
    df.rename(columns={'level_0': 'model', 'level_1': 'dataset'}, inplace=True)

    if write:
        df.to_csv(write, index=False)

    return df


def evaluate_predictions(dirs, model, prefix="peval", class_labels=None, recursive=False, include=None,
                         exclude=None, **kwargs):

    """ Wrapper for evaluating predictions with analysis.predict() on a set of directories containing
    Fast5 files from the labelled classes (species) used for model training. Fast5 files should be independent of
    the ones used for extraction of signal windows for model training. This function returns a confusion matrix
    for assessing prediction errors. """

    fast5 = []
    labels = []
    include, exclude = check_include_exclude(include=include, exclude=exclude, recursive=recursive, verbose=True)
    for label, directory in enumerate(dirs):
        # Recursively grab a list of Fast5 files:
        files = get_recursive_files(directory, recursive=recursive, include=include,
                                    exclude=exclude, extension=".fast5")
        fast5 += files
        labels += [label for _ in files]

    predictions, microseconds, failed_files = predict(fast5=fast5, model=model, **kwargs)

    failed_label_indices = [i for i, file in enumerate(fast5) if file in failed_files]

    df = pandas.DataFrame({
        "file_name": [os.path.basename(file) for file in fast5 if file not in failed_files],
        "label": [label for i, label in enumerate(labels) if i not in failed_label_indices],
        "prediction": predictions,
        "microseconds": microseconds
    })

    fail_df = pandas.DataFrame({
        "file_name": [os.path.basename(file) for file in failed_files]
    })

    # nan = int(df["prediction"].isnull().sum())

    df = df.dropna()

    # print("Removed {} failed prediction from final results.".format(nan))

    fail_df.to_csv(prefix+".fail.txt", index=False)
    df.to_csv(prefix+".csv", index=False)

    cm = confusion_matrix(df["label"], df["prediction"])
    average_prediction_time = df["microseconds"].mean()

    # Normalized confusion matrix:
    cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]

    plot_confusion_matrix(cm, class_labels=class_labels, save=prefix+".pdf", normalize=True)

    return cm, average_prediction_time


def predict(fast5: str or list, model: str, window_max: int = 10, window_size: int = 400, window_step: int = 400,
            window_random: bool = False, scale: bool = False, stdout: bool = True,
            batches=None, mean=False) -> numpy.array:

    """ Predict from Fast5 using loaded model, either from beginning of signal or randomly sampled """

    batch_size = init_batches(batches, window_max)

    # TODO
    if type(fast5) is str and fast5 == "-":
        fast5 = sys.stdin
    elif type(fast5) is list:
        pass
    else:
        raise ValueError("Parameter: fast5 must be list or '-' for stream.")

    print('Loading model...')
    achilles = Achilles()
    achilles.load_model(model_file=model)

    # First pass prediction on null data for initialization:
    print("Initiating model prediction with null data...")
    null_data = numpy.zeros(shape=(1, 1, window_size, 1))
    achilles.predict(null_data)
    print("Starting batch-wise predictions...")

    predicted_labels = []
    prediction_times = []
    failed_files = []
    # TODO does this work with STDIN stream?
    for file_batch in sliding_window(fast5, size=batches, step=batches):

        batch, failed = prepare_batch(file_batch, window_size=window_size, window_step=window_step, normalize=False,
                                      window_random=window_random, window_recover=False, window_max=window_max, scale=scale)

        # Microseconds is per entire batch:
        prediction_windows, microseconds = achilles.predict(batch, batch_size=batch_size)

        # Slice the predictions by window_max and compute mean over slice of batch:
        sliced = prediction_windows.reshape(batch.shape[0]//window_max, window_max, prediction_windows.shape[1])

        # Take the mean of each slice for each label:
        if mean:
            predictions = numpy.mean(sliced, axis=1)
        else:
            # Product, normalized
            predictions = numpy.prod(sliced, axis=1)
            # Normalized product of prediction over slices (len window_max) in batch:
            predictions = numpy.array([norm(prediction) for prediction in predictions])

        # Convert to numeric class labels:
        predicted_label = numpy.argmax(predictions, axis=-1)

        if stdout:
            for i in range(len(predicted_label)):
                print("{}\t{}\t{}".format(predictions[i], predicted_label[i], microseconds))

        predicted_labels += predicted_label.tolist()
        prediction_times += [microseconds for _ in predicted_label]
        failed_files += failed

    return predicted_labels, prediction_times, failed_files


def init_batches(batches, window_max):

    """ Helper function to test parameters and compute batch size based on number of batches and maximum windows """

    batch_size = batches * window_max

    if batch_size % window_max != 0:
        raise ValueError("Batch size ({}) must be a multiple of the number of windows per read ({})."
                         .format(batch_size, window_max))

    print("Batch size per pass through model in Keras:", batch_size)

    return batch_size


def prepare_batch(file_batch, **kwargs):

    """ Helper function to prepare a batch from a list of signal window arrays"""

    # Clean fill-ins from last window and return from iterator:
    file_batch = [file for file in file_batch if file is not None]

    batch = []
    failed = []
    for file in file_batch:
        signal_windows, _ = read_signal(file, **kwargs)

        if signal_windows is not None:
            batch.append(transform_signal_to_tensor(signal_windows))
        else:
            print("Could not read file: ", file)
            failed.append(file)

    batch = numpy.array(batch)

    return batch.reshape(batch.shape[0] * batch.shape[1], batch.shape[2], batch.shape[3], batch.shape[4]), failed

# TODO:


def config_experiments():

    """ Helper function to generate Nextflow configs based on json parameter summaries to run
    lots of different training configurations on GPU in scheduled and replicable manner.
    """

    pass
