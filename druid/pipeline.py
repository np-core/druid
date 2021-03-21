import pandas
from pathlib import Path

class Druid:

    """ Druid pipeline operations """

    def __init__(self, directory: Path):

        self.directory = directory

    def collect_graftm(self):

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
            package_data[package] = pandas.DataFrame(count_data).sort_values("name")

        for p, df in package_data.items():
            print(p)
            print(df)

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



