import ssl

from mlp.cli.cli import cli

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    cli()
