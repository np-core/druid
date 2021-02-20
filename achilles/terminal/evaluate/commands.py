import click
import warnings

from numpy import argmax
from achilles.model import AchillesModel
from achilles.utils import get_dataset_labels
from pathlib import Path


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

    achilles.logger.info(
        f'{seconds:.2f} seconds / {len(predicted)} reads = '
        f'{len(predicted)/seconds:.2f} reads/second'
    )

    predicted = argmax(predicted, -1)

    labels = get_dataset_labels(evaluation)

    correct_labels = 0
    false_labels = 0
    for i, label in enumerate(predicted):
        if int(label) == int(argmax(labels[i])):
            correct_labels += 1
        else:
            false_labels += 1

    achilles.logger.info(f'False predictions in evaluation data: {correct_labels/false_labels:.2f}%')
