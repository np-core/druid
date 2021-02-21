import click
import warnings

from numpy import argmax, where

from achilles.model import AchillesModel
from achilles.utils import get_dataset_labels
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings('ignore')


@click.command()
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model file (HDF5) for prediction",
    show_default=True,
    metavar="",
)
@click.option(
    "--evaluation",
    "-e",
    default=None,
    help="Evaluation file (HDF5) for prediction generator",
    show_default=True,
    metavar="",
)
@click.option(
    "--batch_size",
    "-b",
    default=1000,
    help="Prediction batch size",
    show_default=True,
    metavar="",
)
@click.option(
    "--model_summary",
    "-s",
    is_flag=True,
    help="Print the model summary before prediction",
    show_default=True,
    metavar="",
)
def evaluate(model, evaluation, batch_size, model_summary):

    """ Evaluate a model against a data set from PoreMongo """

    achilles = AchillesModel(evaluation)
    achilles.load_model(model_file=model, summary=model_summary)

    achilles.logger.info(f'Evaluating model: {Path(model).name}')
    achilles.logger.info(f'Using evaluation data from: {Path(evaluation).name}')

    achilles.logger.info(f'Conducting null pass to allocate resources')
    achilles.predict(null_pass=(1, 1, 300, 1), batch_size=batch_size)

    achilles.logger.info(f'Starting predictions with batch size: {batch_size}')
    predicted, microseconds = achilles.predict(data_type="data", batch_size=batch_size)

    seconds = microseconds/1e06
    reads = len(predicted)

    achilles.logger.info(
        f'{seconds:.2f} seconds / {reads} reads = {int(reads/seconds)} reads/second'
    )

    predicted_labels = argmax(predicted, 1)  # one hot decoded
    true_labels = argmax(get_dataset_labels(evaluation), 1)  # one dim, true labels

    cm = confusion_matrix(true_labels, predicted_labels)

    accuracy = accuracy_score(true_labels, predicted_labels)*100
    precision = precision_score(true_labels, predicted_labels)*100
    recall = recall_score(true_labels, predicted_labels)*100
    f1 = f1_score(true_labels, predicted_labels)*100
    roc_auc = roc_auc_score(true_labels, predicted_labels)*100
    achilles.logger.info(
        f"Accuracy: {accuracy:.1f}%  Precision: {precision:.1f}%  Recall: {recall:.2f}%  F1: {f1:.1f}%  ROC-AUC {roc_auc:.1f}%"
    )

