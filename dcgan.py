import keras
import numpy as np
import tensorflow as tf
import time
import random
from IPython import display

from utils import generate_and_save_images, Paths
from models import make_generator_model, make_discriminator_model


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
        self.seed = tf.random.normal([self.batch_size, self.noise_dim, self.noise_dim, 3])
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
        noise = tf.random.normal([self.batch_size, self.noise_dim, self.noise_dim, 3])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(images, training=True)
            gen_loss = self.generator_loss(generated_images, images)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

    def train(self):
        for epoch in range(self.epochs):
            start = time.time()
            counter = 0
            for image_batch in self.train_dataset:
                print("yess")
                self.train_step(image_batch)
                counter += 1
                print(counter)

            # Produce images for the GIF as you go
            display.clear_output(wait=True)
            generate_and_save_images(
                self.generator,
                epoch + 1,
                self.seed,
                show=True
            )
            # Save the model every 15 epochs
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix = self.create_train_checkpoint_directory)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

