from abc import abstractmethod
from typing import Any
from keras import *
import tensorflow as tf
from tensorflow.python.data.experimental.ops.data_service_ops import distribute

from utils import Params

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
    def __init__(
            self,
            params: Params,
    ):
        self.params = params
        self.unit = params.get('unit')
        self.frame_size = params.get('image_size')
        self.noise_dimension = params.get('noise_dimension')
        self.iteration = params.get('generator_layer_iterator')
        self.model: Sequential = Sequential()
        self.divide = 2**params.get('generator_layer_iterator')
        self.kernel = params.get('kernel')
        self.layer_strides = params.get('generator_stride')
        self.loss_function = losses.BinaryCrossentropy(from_logits=True)

    @classmethod
    def  make_model(cls, params: Params):
        generator = Generator(
            params
        )
        generator.build()
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


class Discriminator:
    def __init__(
            self,
            params: Params,
    ):
        self.params = params
        self.unit = params.get('unit')
        self.frame_size = params.get('image_size')
        self.noise_dimension = params.get('noise_dimension')
        self.iteration = params.get('discriminator_layer_iterator')
        self.model: Sequential = Sequential()
        self.divide = 2**params.get('discriminator_layer_iterator')
        self.kernel = params.get('kernel')
        self.stride = params.get('discriminator_stride')
        self.dropout_ratio = params.get('dropout_ratio')
        self.loss_function = losses.BinaryCrossentropy(from_logits=True)

    @classmethod
    def make_model(cls, params: Params):
        discriminator = Discriminator(
            params
        )
        discriminator.build()
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
