import click
import warnings

from numpy import argmax
from achilles.model import AchillesModel
from achilles.utils import get_dataset_labels
from colorama import Fore
from pathlib import Path

Y = Fore.YELLOW
G = Fore.GREEN
RE = Fore.RESET

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

    print(f'{Y}Evaluating model: {G}{Path(model).name}{RE}')
    print(f'{Y}Using evaluation data from: {G}{Path(evaluation).name}{RE}')

    print(f'{Y}Conducting null pass to allocate resources on {G}GPU ...{RE}')
    achilles.predict(null_pass=(1, 1, 300, 1), batch_size=batch_size)

    print(f'{Y}Starting predictions ...{RE}')
    predicted, microseconds = achilles.predict(data_type="data", batch_size=batch_size)

    seconds = microseconds/1e06
    print(predicted)

    print(f'Prediction speed: {seconds:.2f} seconds = {len(predicted)/seconds:.2f} reads/second')

    predicted = argmax(predicted, -1)

    labels = get_dataset_labels(evaluation)

    correct_labels = 0
    false_labels = 0
    for i, label in enumerate(predicted):
        if int(label) == int(argmax(labels[i])):
            correct_labels += 1
        else:
            false_labels += 1

    print(f'False predictions in evaluation data: {correct_labels/false_labels:.2f}%')
