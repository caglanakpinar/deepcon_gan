from abc import abstractmethod
from typing import Any
from keras import *
import tensorflow as tf
from tensorflow.python.data.experimental.ops.data_service_ops import distribute

cross_entropy = losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = optimizers.Adam(1e-4)
discriminator_optimizer = optimizers.Adam(1e-4)


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



class Generator:
    def __init__(self, unit, frame_size, noise_dimension, generator_layer_iterator, kernel, generator_strides: list[int]=[1, 2, 2]):
        self.unit = unit
        self.frame_size = frame_size
        self.noise_dimension = noise_dimension
        self.iteration = generator_layer_iterator
        self.model: Sequential = Sequential()
        self.divide = 2**generator_layer_iterator
        self.kernel = kernel
        self.layer_strides = generator_strides
        self.loss_function = losses.BinaryCrossentropy(from_logits=True)

    @classmethod
    def  make_model(cls, unit, frame_size, noise_dimension, generator_layer_iterator, kernel, generator_strides: list[int]=[1, 2, 2], **kwargs):
        generator = Generator(
            unit, frame_size, noise_dimension, generator_layer_iterator, kernel, generator_strides
        )
        generator.build()
        print(generator.model.summary())
        return generator.model

    @staticmethod
    def loss(fake_output):
        return losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_output), fake_output)

    def batch_norm_and_l_relu(self):
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU())

    def build(self):
        self.model.add(layers.Input((self.noise_dimension,)))
        self.model.add(layers.Dense(int(self.frame_size[0] / self.divide) * int(self.frame_size[1] / self.divide) * self.unit, use_bias=False))
        self.batch_norm_and_l_relu()
        self.model.add(layers.Reshape((int(self.frame_size[0] / self.divide), int(self.frame_size[1] / self.divide), self.unit)))
        for i, stride in zip(range(1, self.iteration+1), self.layer_strides):
            self.model.add(
                layers.Conv2DTranspose(
                    int(self.unit / (2**i)),
                    (self.kernel, self.kernel),
                    strides=(stride, stride),
                    padding='same',
                    use_bias=False
                )
            )
            self.batch_norm_and_l_relu()
        self.model.add(
            layers.Conv2DTranspose(
                self.frame_size[2],
                kernel_size=(5, 5),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                activation="tanh"
            )
        )


def make_generator_model(unit, frame_size, noise_dimension) -> Sequential:
    model = Sequential()
    model.add(layers.Input((noise_dimension, )))
    model.add(layers.Dense(int(frame_size[0]/8)*int(frame_size[0]/8)*unit, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((int(frame_size[0]/8), int(frame_size[0]/8), unit)))
    assert model.output_shape == (None, int(frame_size[0]/8), int(frame_size[0]/8), unit)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(int(unit/2), (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, int(frame_size[0]/8), int(frame_size[0]/8), int(unit/2))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(int(unit/4), (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(frame_size[0]/4), int(frame_size[0]/4), int(unit/4))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(int(unit/8), (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, int(frame_size[0]/2), int(frame_size[0]/2), int(unit/8))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=(2, 2), padding='same', use_bias=False, activation="tanh"))
    assert model.output_shape == (None, frame_size[0], frame_size[1], 3)
    model.summary()
    return model


class Discriminator:
    def __init__(self, unit, frame_size, noise_dimension, discriminator_layer_iterator, kernel, dropout_ratio, discriminator_strides):
        self.unit = unit
        self.frame_size = frame_size
        self.noise_dimension = noise_dimension
        self.iteration = discriminator_layer_iterator
        self.model: Sequential = Sequential()
        self.divide = 2**discriminator_layer_iterator
        self.kernel = kernel
        self.stride = discriminator_strides
        self.dropout_ratio = dropout_ratio
        self.loss_function = losses.BinaryCrossentropy(from_logits=True)

    @classmethod
    def make_model(cls, unit, frame_size, noise_dimension, discriminator_layer_iterator, kernel, dropout_ratio, discriminator_strides, **kwargs):
        discriminator = Discriminator(
            unit, frame_size, noise_dimension, discriminator_layer_iterator, kernel, dropout_ratio, discriminator_strides
        )
        discriminator.build()
        print(discriminator.model.summary())
        return discriminator.model

    @staticmethod
    def loss(real_output, fake_output):
        real_loss = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
        fake_loss = losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def l_relu_dropout(self):
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(self.dropout_ratio))

    def build(self):
        self.model.add(layers.Input((self.frame_size[0], self.frame_size[1], 3, )))
        for i in range(self.iteration):
            self.model.add(
                layers.Conv2D(
                    int(self.unit*(2**i)),
                    (self.kernel, self.kernel),
                    strides=(self.stride, self.stride),
                    padding='same',
                )
            )
            self.l_relu_dropout()
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))


def make_discriminator_model(unit, frame_size) -> Sequential:
    model = Sequential()
    model.add(layers.Conv2D(unit, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[frame_size[0], frame_size[1], 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(int(unit*2), (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(int(unit*4), (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(int(unit*8), (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.summary()
    return model
