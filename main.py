import glob
from email.policy import default

import imageio
import click
import os
import time
from IPython import display
from math import sqrt
import ssl

from dcgan import DCGAN
from models import *
from utils import *


@click.group()
def cli():
    pass

ssl._create_default_https_context = ssl._create_unverified_context

BUFFER_SIZE = 10000
BATCH_SIZE = 128
RESIZE_IMAGE_FRAME_SIZE = (64, 64)
TRANSPOSE_ITERATOR = 4
# unit will be 30
UNIT = 32

EPOCHS = 50
noise_dim = 64
num_examples_to_generate = 1
TRANSPOSE_ITERATOR = 2


# (train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# generator = make_generator_model(30 , BATCH_SIZE)
# noise = tf.random.normal([1, 100])
# generated_image = generator(noise, training=False)
# discriminator = make_discriminator_model(32)
# cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator)
#


# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
# seed = tf.random.normal([num_examples_to_generate, noise_dim])





# # Notice the use of `tf.function`
# # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(images):
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])
#
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#       generated_images = generator(noise, training=True)
#
#       real_output = discriminator(images, training=True)
#       fake_output = discriminator(generated_images, training=True)
#
#       gen_loss = generator_loss(fake_output)
#       disc_loss = discriminator_loss(real_output, fake_output)
#
#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#

# def train(dataset, epochs):
#   for epoch in range(epochs):
#     start = time.time()
#
#     for image_batch in dataset:
#       train_step(image_batch)
#
#     # Produce images for the GIF as you go
#     display.clear_output(wait=True)
#     generate_and_save_images(generator,
#                              epoch + 1,
#                              seed)
#     # Save the model every 15 epochs
#     if (epoch + 1) % 15 == 0:
#       checkpoint.save(file_prefix = checkpoint_prefix)
#     print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

@cli.command()
@click.option(
    "--name",
    default="caglan",
    help="folder name to capture images"
)
def capture_images_and_train(
    name,
):
    ci = CapturingImages(BATCH_SIZE, 32)
    ci.capture_images(name=name, buffer_size=BUFFER_SIZE, frame_size=RESIZE_IMAGE_FRAME_SIZE)
    images = ci.read_images(name, RESIZE_IMAGE_FRAME_SIZE)
    model = DCGAN(
        generator=make_generator_model(UNIT, BATCH_SIZE, RESIZE_IMAGE_FRAME_SIZE, noise_dim),
        generator_optimizer=generator_optimizer,
        generator_loss=generator_loss,
        discriminator=make_discriminator_model(UNIT, RESIZE_IMAGE_FRAME_SIZE),
        discriminator_optimizer=discriminator_optimizer,
        discriminator_loss=discriminator_loss,
        batch_size=BATCH_SIZE,
        noise_dim=noise_dim,
        epochs=EPOCHS,
        train_images=images.images
    )
    model.train()


if __name__ == '__main__':
    cli()
