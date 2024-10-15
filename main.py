import click
import ssl

from dcgan import DCGAN
from utils import *
from configs import *


ssl._create_default_https_context = ssl._create_unverified_context


@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--trainer_config_path",
    default="configs",
    help="where trainer .yaml is being stored"
)
def capture_images_and_train(
    trainer_config_path
):
    params = Params(trainer_config_path)
    ci = CapturingImages.read(
        params
    )
    model = DCGAN(
        params
    )
    model.train(ci.images)


@cli.command()
@click.option(
    "--trainer_config_path",
    default="configs/params.yaml",
    help="where trainer .yaml is being stored"
)
def train_with_checkpoint(
    trainer_config_path,
):
    params = Params(trainer_config_path)
    ci = CapturingImages.read(
        params
    )
    model = DCGAN.read_checkpoint(params)
    model.train(ci.images)



if __name__ == '__main__':
    cli()
