import pandas
import math
from pathlib import Path

from matplotlib import pyplot as plt
import seaborn as sns

class DruidPipeline:

    """ Druid pipeline operations """

    def __init__(self, directory: Path):

        self.directory = directory

    def collect_graftm(self) -> dict:

        graftm_path = self.directory / 'graftm'

        package_paths = [package for package in graftm_path.glob("*") if package.is_dir()]

        sample_paths = {
            package: [sample for sample in package.glob("*") if package.is_dir()]
            for package in package_paths
        }

        counts = {}
        for package, samples in sample_paths.items():
            if package not in counts.keys():
                counts[package] = []

            for sample_path in samples:
                root, bacteria, archaea = self.process_graftm_counts(file=sample_path / "combined_count_table.txt")
                sample_data = {'name': sample_path.name, 'root': root, 'bacteria': bacteria, 'archaea': archaea}
                counts[package].append(sample_data)

        package_data = {}
        for package, count_data in counts.items():
            package_data[package] = pandas.DataFrame(count_data).sort_values("name").reset_index().melt(
                id_vars=['name'], value_vars=['root', 'bacteria', 'archaea'], var_name='tax', value_name='reads'
            )
            print(package_data[package])

        return package_data

    def plot_graftm_counts(self, package_data: dict, plot_name: str):

        nrow, ncol = math.ceil(len(package_data) / 2), 2

        fig, axes = plt.subplots(
            nrows=nrow, ncols=ncol, figsize=(
                nrow * 7, ncol * 4.5
            )
        )

        for i, (package, df) in enumerate(package_data.items()):
            r, c = self._get_axes_idx(i, ncol)
            print(i, r, c)
            sns.barplot(
                data=df, ax=axes[r, c] if nrow > 1 else axes[i],
                x="name", y="reads", hue="tax", palette="dark", alpha=.6
            )

        plt.tight_layout()

        fig.savefig(f'{plot_name}.pdf')
        fig.savefig(f'{plot_name}.svg')
        fig.savefig(f'{plot_name}.png')

    @staticmethod
    def process_graftm_counts(file: Path):

        """ Process the operon counts into Archaea / Bacteria / Root """

        df = pandas.read_csv(file, sep="\t", header=None, names=["IDX", "READS", "TAX"], comment="#")

        root = 0
        bacteria = 0
        archaea = 0
        for i, row in df.iterrows():
            reads = df.at[i, "READS"]
            if row["TAX"] == "Root":
                root += reads
            if "k__Bacteria" in row["TAX"]:
                bacteria += reads
            if "k__Archaea" in row["TAX"]:
                archaea += reads

        return root, bacteria, archaea

    @staticmethod
    def _get_axes_idx(i, ncol):

        if i == 0:
            r, c = 0, 0
        else:
            r = math.floor(i / ncol)
            c = i % ncol

        return r, c


