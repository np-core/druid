import click
import pandas

from pathlib import Path
from matplotlib import pyplot as plt

@click.command()
@click.option(
    "--evaluation_path",
    "-e",
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
    default="tab20",
    type=str,
    help="Matplotlib color map for unique color per log file",
    show_default=True,
    metavar="",
)
@click.option(
    "--detail",
    "-d",
    is_flag=True,
    help="Output additional plots: AUC ROC",
    show_default=True,
    metavar="",
)
def plot_evaluation(log_path, plot_file, color, detail):

    """Plot training accuracy and loss """

    if log_path.is_dir():
        logs = list(Path(log_path).glob("**/*.log"))
    else:
        raise ValueError(f"Could not find log path or file: {log_path}")

    if not logs:
        print(f"Could not find log files (.log) at: {log_path}")
        exit(1)

    with plt.style.context('seaborn-white'):
        f, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 4.5))

        params = {
            'legend.fontsize': 6, 'axes.labelsize': 10
        }

        plt.rcParams.update(params)

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.tick_params(axis='both', labelsize=8, length=2, width=2)
            ax.set_xlabel('\nEpochs')

        cm = plt.get_cmap(color)
        cmp = [cm(1. * i / len(log_data)) for i in range(len(log_data))]

        axes[0].set_prop_cycle(color=cmp)
        axes[1].set_prop_cycle(color=cmp)

        axes[0].set_ylabel('Accuracy\n')
        axes[1].set_ylabel('Loss\n')

        # Plot

        plt.tight_layout()
        plt.savefig(plot_file)