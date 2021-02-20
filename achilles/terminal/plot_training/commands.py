import click
import pandas

from pathlib import Path
from matplotlib import pyplot as plt

@click.command()
@click.option(
    "--log_path",
    "-l",
    default="local",
    type=Path,
    help="Path to directory containing one or multiple model training directories",
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
def plot_training(log_path, plot_file, color):

    """Plot training accuracy and loss """

    if log_path.is_dir():
        logs = list(Path(log_path).glob("**/*.log"))
    else:
        raise ValueError(f"Could not find log path or file: {log_path}")

    if not logs:
        print(f"Could not find log files (.log) at: {log_path}")
        exit(1)

    log_data = []
    for f in logs:
        d = pandas.read_csv(f, sep=",")
        d['name'] = [f.parent.name for _ in d.iterrows()]
        log_data.append(d)

    if log_data:
        df = pandas.concat(log_data)
    else:
        raise ValueError(f"Could not parse log data from: {log_path}")

    df_group = df.groupby("name")

    with plt.style.context('seaborn-white'):
        f, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 4.5))

        params = {
            'legend.fontsize': 6
        }

        plt.rcParams.update(params)

        for ax in axes:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        plt.tick_params(axis='both', labelsize=1, length=1)

        cm = plt.get_cmap(color)
        cmp = [cm(1. * i / len(log_data)) for i in range(len(log_data))]

        axes[0].set_prop_cycle(color=cmp)
        axes[1].set_prop_cycle(color=cmp)

        df_group["accuracy"].plot(x="epochs", legend=True, ax=axes[0], title="Training accuracy")
        df_group["loss"].plot(x="epochs", legend=False, ax=axes[1], title="Training loss")

        plt.tight_layout()
        plt.savefig(plot_file)