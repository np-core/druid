import click
import warnings
import pandas

from numpy import argmax, where, split, product, array

from achilles.model import AchillesModel
from achilles.utils import get_dataset_labels
from pathlib import Path
from colorama import Fore

from sklearn.metrics import \
    confusion_matrix, \
    precision_score, \
    accuracy_score, \
    recall_score, \
    f1_score, \
    roc_auc_score

warnings.filterwarnings('ignore')

G = Fore.GREEN
Y = Fore.YELLOW
RE = Fore.RESET


@click.command()
@click.option(
    "--model",
    "-m",
    default=None,
    type=Path,
    help="Model file (HDF5) for prediction",
    show_default=True,
    metavar="",
)
@click.option(
    "--evaluation",
    "-e",
    default=None,
    type=Path,
    help="Evaluation file (HDF5) for prediction generator",
    show_default=True,
    metavar="",
)
@click.option(
    "--training_path",
    "-tp",
    default=None,
    type=Path,
    help="Path to training path containing dirs with model files (HDF5) for pairwise comparison",
    show_default=True,
    metavar="",
)
@click.option(
    "--evaluation_path",
    "-ep",
    default=None,
    type=Path,
    help="Path to evaluation files (HDF5) for pairwise comparison",
    show_default=True,
    metavar="",
)
@click.option(
    "--batch_size",
    "-bs",
    default=1000,
    help="Prediction batch size",
    show_default=True,
    metavar="",
)
@click.option(
    "--slice",
    "-sc",
    default=None,
    type=int,
    help="Prediction on the product of <slice> overlapping windows; HDF5 evaluatio nfile must be ordered as such",
    show_default=True,
    metavar="",
)
@click.option(
    "--model_summary",
    "-ms",
    is_flag=True,
    help="Print the model summary before prediction",
    show_default=True,
    metavar="",
)
def evaluate(model, evaluation, training_path, evaluation_path, batch_size, model_summary, slice):

    """ Evaluate a model against a data set from PoreMongo """

    if not training_path and not evaluation_path:
        # Single evaluation of model
        run_evaluation(model=model, evaluation=evaluation, slice=slice, batch_size=batch_size, model_summary=model_summary)
    else:
        # Pairwise models and evaluation sets
        model_files = [
            list(p.glob(f'*.checkpoint.val_loss.h5'))[0]
            for p in training_path.glob("*/") if p.is_dir()
        ]
        evaluation_files = [p for p in evaluation_path.glob("*.hdf5") if p.is_file()]

        rows = [
            run_evaluation(
                model=model,
                evaluation=evaluation,
                batch_size=batch_size
            )
            for model in model_files
            for evaluation in evaluation_files
        ]

        results = pandas.DataFrame(
            rows, columns=[
                'model', 'eval', 'batch_size', 'reads', 'sec', 'rs',
                'accuracy', 'precision', 'recall', 'f1', 'roc-auc',
                'tp', 'fp', 'tn', 'fn'
            ]
        )

        results.to_csv('data.tsv', sep='\t', index=False, header=True)


def run_evaluation(model: Path, evaluation: Path, slice: int = None, batch_size: int = 5000, model_summary: bool = False):

    """ Evaluate a model against a data set from PoreMongo """

    achilles = AchillesModel(evaluation)
    input_shape = achilles.load_model(model_file=model, summary=model_summary)

    achilles.logger.info(f'Evaluating model: {Y}{model.parent.name}{RE}')
    achilles.logger.info(f'Using evaluation data from: {Y}{evaluation.name}{RE}')

    achilles.logger.info(f'Conducting null pass to allocate resources')
    achilles.predict(null_pass=(1, 1, input_shape[2], 1), batch_size=batch_size)

    achilles.logger.info(f'Starting predictions with batch size: {Y}{batch_size}{RE}')
    predicted, microseconds = achilles.predict(data_type="data", batch_size=batch_size)

    seconds = microseconds/1e06
    reads = len(predicted)

    achilles.logger.info(
        f'{seconds:.2f} seconds / {reads} reads = {int(reads/seconds)} reads/second'
    )

    if slice is not None:
        predicted_slices = split(
            predicted, [i for i in range(len(predicted)) if i % slice == 0], axis=0
        )
        print(predicted_slices)
        print(predicted_slices.shape)
        predicted_probability = product(predicted_slices, axis=2)
        print(predicted_probability)
        predicted_labels = argmax(predicted_probability, 1)
    else:
        predicted_labels = argmax(predicted, 1)  # one hot decoded

    true_labels = argmax(get_dataset_labels(evaluation), 1)  # one dim, true labels

    # Binary case!
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    roc_auc = roc_auc_score(true_labels, predicted_labels)

    achilles.logger.info(
        f"{Y}Accuracy: {G}{accuracy:.3f}  {Y}Precision: {G}{precision:.3f}  "
        f"{Y}Recall: {G}{recall:.3f}  {Y}F1: {G}{f1:.3f}  {Y}ROC-AUC {G}{roc_auc:.3f}{RE}"
    )
    achilles.logger.info(
        f"True positives: {tn}  True negatives: {tn}  False positives: {fp}  False negatives: {fn}"
    )

    return (
        model.parent.stem, evaluation.stem, batch_size, reads, seconds, int(reads/seconds),
        accuracy, precision, recall, f1, roc_auc, tp, fp, tn, fn
    )