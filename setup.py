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
        "scikit-image",
        "scikit-learn",
        "h5py",
        "scipy",
        "tqdm",
        "seaborn",
        "matplotlib",
        "colorama",
        "pymongo",
        "mongoengine",
        "keras",
        "click",
        "pyyaml",
        "pytest",
        "wget",
        "watchdog",
        "paramiko",
        "numpy",
        "tqdm",
        "colorama",
        "pymongo",
        "mongoengine",
        "ont_fast5_api",
        "pandas",
        "paramiko",
        "scp",
        "scikit-image",
        "scipy",
        "watchdog",
        "apscheduler",
        "click",
        "deprecation",
        "pyfastaq",
        "drep"
        "tensorflow"
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
