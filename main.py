import click
import ssl

from dcgan import DCGAN
from models import *
from utils import *
from configs import *


ssl._create_default_https_context = ssl._create_unverified_context


@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--name",
    default="caglan",
    help="folder name to capture images"
)
def capture_images_and_train(
    name,
):
    ci = CapturingImages(64, 32)
    if not (Path(__file__).absolute().parent / name).exists():
        ci.capture_images(name=name, buffer_size=BUFFER_SIZE, frame_size=RESIZE_IMAGE_FRAME_SIZE)
    ci = ci.read_images(name, RESIZE_IMAGE_FRAME_SIZE)
    model = DCGAN(
        generator=make_generator_model(UNIT, RESIZE_IMAGE_FRAME_SIZE, noise_dim),
        generator_optimizer=generator_optimizer,
        generator_loss=generator_loss,
        discriminator=make_discriminator_model(UNIT, RESIZE_IMAGE_FRAME_SIZE),
        discriminator_optimizer=discriminator_optimizer,
        discriminator_loss=discriminator_loss,
        batch_size=BATCH_SIZE,
        noise_dim=noise_dim,
        epochs=EPOCHS,
        train_images=ci.images
    )
    model.train()


if __name__ == '__main__':
    cli()
