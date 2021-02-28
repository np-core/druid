import click
import pandas
import seaborn as sn
from pathlib import Path
from matplotlib import pyplot as plt


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
        f, axes = plt.subplots(
            nrows=3, ncols=2, figsize=(14*3, 14*2)
        )

        for (i, row) in enumerate(axes):
            for j, _ in enumerate(row):
                try:
                    metric = matrices[i]
                except IndexError:
                    continue

                d = create_data_matrix(data_frame=df, column=metric)
                d.index.name = 'Model'
                d.columns.name = 'Evaluation'
                sn.set(font_scale=1.4)  # for label size
                sn.heatmap(d, cmap=color, annot=True, annot_kws={"size": 16}, ax=axes[i][j])  # font size

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
