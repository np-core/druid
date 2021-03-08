import click

VERSION = "0.3-alpha"


@click.group()
@click.version_option(version=VERSION)
def utils():
    pass

