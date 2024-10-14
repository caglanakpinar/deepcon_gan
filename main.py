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
    "--name",
    default="caglan",
    help="folder name to capture images"
)
def capture_images_and_train(
    name,
):
    params = Params.read_from_config("params", "configs")
    ci = CapturingImages.read_images(
        name=name, batch_size=BATCH_SIZE, image_size=RESIZE_IMAGE_FRAME_SIZE, buffer_size=BUFFER_SIZE
    )
    model = DCGAN(
        name=name,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        noise_dimension=NOISE_DIMENSION,
        unit=UNIT,
        frame_size=RESIZE_IMAGE_FRAME_SIZE,
        generator_layer_iterator=GENERATOR_LAYER_ITERATOR,
        discriminator_layer_iterator= DISCRIMINATOR_LAYER_ITERATOR,
        kernel=KERNEL,
        generator_strides=GENERATOR_STRIDE,
        discriminator_strides=DISCRIMINATOR_STRIDE,
        dropout_ratio=DROPOUT_RATIO,
        lr_generator=LR_GENERATOR,
        lr_discriminator=LR_DISCRIMINATOR
    )
    model.train(ci.images)


@click.group()
def cli():
    pass

@cli.command()
@click.option(
    "--name",
    default="caglan",
    help="folder name to capture images"
)
def train_with_checkpoint(
    name,
):
    params = Params.read_from_config("params", "configs")
    ci = CapturingImages.read_images(
        name=name, batch_size=BATCH_SIZE, image_size=RESIZE_IMAGE_FRAME_SIZE, buffer_size=BUFFER_SIZE
    )
    model = DCGAN.read_checkpoint(
        name=name,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        noise_dimension=NOISE_DIMENSION,
        unit=UNIT,
        frame_size=RESIZE_IMAGE_FRAME_SIZE,
        generator_layer_iterator=GENERATOR_LAYER_ITERATOR,
        discriminator_layer_iterator= DISCRIMINATOR_LAYER_ITERATOR,
        kernel=KERNEL,
        generator_strides=GENERATOR_STRIDE,
        discriminator_strides=DISCRIMINATOR_STRIDE,
        dropout_ratio=DROPOUT_RATIO,
        lr_generator=LR_GENERATOR,
        lr_discriminator=LR_DISCRIMINATOR
    )
    model.train(ci.images)



if __name__ == '__main__':
    cli()
