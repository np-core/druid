from setuptools import setup, find_packages

setup(
    name="druid",
    url="https://github.com/esteinig/druid",
    author="Eike Steinig",
    author_email="eike.steinig@unimelb.edu.au",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "h5py",
        "scipy",
        "seaborn",
        "matplotlib",
        "colorama",
        "pymongo",
        "mongoengine",
        "keras",
        "pyyaml",
        "pytest",
        "wget",
        "watchdog",
        "numpy",
        "tqdm",
        "ont_fast5_api",
        "scp",
        "scikit-image",
        "apscheduler",
        "click",
        "deprecation",
        "pyfastx",
        "drep",
        "tensorflow",
        "pyvista",
        "pdb2pqr"
    ],
    entry_points="""
        [console_scripts]
        druid=druid.terminal.client:terminal_client
    """,
    version="0.4",
    license="MIT",
    description="Druid is a platform to train and evaluate neural networks for "
                "nanopore signal classification of non-canonical base modifications",
)
