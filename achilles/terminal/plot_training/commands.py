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
def plot_training(log_path, plot_file):

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

    f, axes = plt.subplots(2, 2)
    f.suptitle("Training Summary")

    df_group["accuracy"].plot(x="epochs", legend=True, ax=axes[0][0], title="Training accuracy")
    df_group["loss"].plot(x="epochs", legend=False, ax=axes[0][1], title="Training loss")
    df_group["val_accuracy"].plot(x="epochs", legend=False, ax=axes[1][0], title="Validation accuracy")
    df_group["val_loss"].plot(x="epochs", legend=False, ax=axes[1][1], title="Validation loss")

    plt.tight_layout()
    plt.savefig(plot_file, figsize=(8, 6), bbox_inches="tight")