import click
import ssl

from dcgan import DCGAN, HyperDCGAN
from utils import *


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


@cli.command()
@click.option(
    "--trainer_config_path",
    default="configs/params.yaml",
    help="where trainer .yaml is being stored"
)
@click.option(
    "--hyper_parameter_config_path",
    default="configs/hyper_params.yaml",
    help="where hyperparameter tuning .yaml is being stored"
)
def hyper_parameters(
    trainer_config_path,
    hyper_parameter_config_path,
):
    params = Params(trainer_config_path)
    hyper_params = Params(hyper_parameter_config_path)
    ci = CapturingImages.read(
        params
    )
    hyper_model = HyperDCGAN()
    hyper_model.set_model(DCGAN)
    hyper_model.set_train_params(params)
    hyper_model.set_hyper_params(hyper_params)
    hyper_model.random_search(
        hyper_model,
        x=ci.images, y=ci.images,  validation_data=ci.images,
        max_trials=hyper_params.get('max_trials')
    )


if __name__ == '__main__':
    cli()
