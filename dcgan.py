import keras
import numpy as np
import tensorflow as tf
import time
from IPython import display
import matplotlib.pyplot as plt

from utils import Paths
from models import make_generator_model, make_discriminator_model, Generator, Discriminator


class DCONGAN(Paths):
    def __init__(self, batch_size, noise_dim, unit, noise_dimension, frame_size, kernel, opt, lr, dropout_ratio):
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.generator = Generator(
            unit, noise_dimension, frame_size, kernel, opt, lr
        )
        self.discriminator = Discriminator(
            unit, noise_dimension, frame_size, kernel, opt, lr, dropout_ratio
        )
        self.generator_optimizer = self.generator.optimizer()
        self.discriminator_optimizer = self.discriminator.optimizer()


    @property
    def checkpoint(self) -> tf.train.Checkpoint:
        return tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator.loss(fake_output)
            disc_loss = self.discriminator.loss((real_output, fake_output))

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            for image_batch in self.train_dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(
                epoch + 1,
                self.seed,
                show=True
            )
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.create_train_checkpoint_directory)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    def generate_and_save_images(self, epoch, test_input, show=False):
        predictions = self.generator(test_input, training=False)
        plt.imshow(((predictions[0].numpy() * 127.5) + 127.5).astype(int))
        plt.axis('off')

        plt.savefig(self.create_train_checkpoint_directory / 'image_at_epoch_{:04d}.png'.format(epoch))
        if show:
            plt.show()



class DCGAN(Paths):
    def __init__(
        self,
        generator: make_generator_model,
        discriminator: make_discriminator_model,
        generator_optimizer: keras.optimizers.Adam,
        discriminator_optimizer,
        generator_loss,
        discriminator_loss,
        batch_size: int,
        noise_dim: int,
        epochs: int,
        train_images: np.ndarray
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.seed = tf.random.normal([1, self.noise_dim])
        self.epochs = epochs
        self.train_dataset = (
            tf.data.Dataset
            .from_tensor_slices(train_images)
            .shuffle(train_images.shape[0]).batch(self.batch_size)
        )

    @property
    def checkpoint(self) -> tf.train.Checkpoint:
        return tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            for image_batch in self.train_dataset:
                self.train_step(image_batch)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            self.generate_and_save_images(
                epoch + 1,
                self.seed,
                show=True
            )
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.create_train_checkpoint_directory)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    def generate_and_save_images(self, epoch, test_input, show=False):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator(test_input, training=False)
        plt.imshow(((predictions[0].numpy() * 127.5) + 127.5).astype(int))
        plt.axis('off')

        plt.savefig(self.create_train_checkpoint_directory / 'image_at_epoch_{:04d}.png'.format(epoch))
        if show:
            plt.show()

