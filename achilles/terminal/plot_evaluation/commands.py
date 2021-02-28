import click
import pandas
import seaborn as sn
import numpy as np

from pathlib import Path
from matplotlib import pyplot as plt
from achilles.utils import carto_tropical_diverging

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
    "--remove",
    "-r",
    default="ecoli2",
    type=str,
    help="Remove any training or evaluation files that contain this string [ecoli2]",
    show_default=True,
    metavar="",
)
def plot_evaluation(data, plot_file, remove):

    """Plot training accuracy and loss """

    df = pandas.read_csv(data, sep="\t", header=0)

    rows = []
    for i, row in df.iterrows():
        if remove in row['model'] or remove in row['eval']:
            continue
        else:
            rows.append(row)

    df = pandas.DataFrame(rows, columns=df.columns.tolist())

    print(f'Average prediction speed: {df.ws.mean():.2f} windows / second')

    matrices = [['accuracy', 'precision'], ['recall', 'f1'], ['roc-auc']]

    cm = carto_tropical_diverging(reverse=True)

    with plt.style.context('seaborn-white'):
        f, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(14*3, 14*2)
        )

        for (i, row) in enumerate(axes):
            for j, _ in enumerate(row):
                try:
                    metric = matrices[i][j]
                except IndexError:
                    continue

                d = create_data_matrix(data_frame=df, column=metric)
                d.index.name = 'Model'
                d.columns.name = 'Evaluation'
                sn.set(font_scale=1.4)  # for label size
                sn.heatmap(d, cmap=cm, annot=True, annot_kws={"size": 16}, ax=axes[i][j], linewidths=5)  # font size
                axes[i][j].set_title(metric.capitalize() if not metric == 'roc-auc' else metric.upper())

    plt.tight_layout()
    plt.savefig(plot_file)


def create_data_matrix(data_frame: pandas.DataFrame, column: str = "accuracy"):

    # dataframes of pairwise evaluation are ordered

    dm = []
    column_labels = []
    row_labels = []
    for (i, (model, dt)) in enumerate(data_frame.groupby('model')):
        row_labels.append(model)
        dm.append(
            dt[column].tolist()
        )
        if i == 0:
            column_labels = dt['eval'].tolist()

    df_cm = pandas.DataFrame(dm, columns=column_labels, index=row_labels)

    return df_cm
