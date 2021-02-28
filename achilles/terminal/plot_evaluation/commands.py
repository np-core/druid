import click
import pandas
import numpy as np
import seaborn as sn
from pathlib import Path
from matplotlib import pyplot as plt

from achilles.utils import plot_confusion_matrix


@click.command()
@click.option(
    "--data",
    "-d",
    default="local",
    type=Path,
    help="Path to directory containing the evaluation directories",
    show_default=True,
    metavar="",
)
@click.option(
    "--plot_file",
    "-p",
    default="training.pdf",
    type=Path,
    help="Output plot file, extension determines file type",
    show_default=True,
    metavar="",
)
@click.option(
    "--color",
    "-c",
    default="Greens",
    type=str,
    help="Color palette for confusion heatmaps",
    show_default=True,
    metavar="",
)
def plot_evaluation(data, plot_file, color):

    """Plot training accuracy and loss """

    df = pandas.read_csv(data, sep="\t", header=0)

    print(f'Average prediction speed: {df.ws.mean():.2f} windows / second')

    matrices = ('accuracy', 'precision', 'recall', 'f1', 'roc-auc')

    with plt.style.context('seaborn-white'):
        f, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 4.5))

        for i, ax in enumerate(axes):
            dm, labels = create_data_matrix(data_frame=df, column=matrices[i])
            df_cm = pandas.DataFrame(dm, columns=labels, index=labels)
            print(df_cm)
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            sn.set(font_scale=1.4)  # for label size
            sn.heatmap(df_cm, cmap=color, annot=True, annot_kws={"size": 16}, ax=ax)  # font size

    plt.tight_layout()
    plt.savefig(plot_file)


def create_data_matrix(data_frame: pandas.DataFrame, column: str = "accuracy"):

    dm = []
    labels = []  # dataframes of pairwise evaluation are ordered
    for model, dt in data_frame.groupby('model'):
        labels.append(model)
        dm.append(
            dt[column].tolist()
        )

    return dm, labels
