from abc import abstractmethod
from typing import Any

from keras import *
import tensorflow as tf

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


class BaseModel(Model):
    def __init__(self):
        super().__init__()
        self.cross_entropy_B = losses.BinaryCrossentropy(from_logits=True)
        self.optimizers = {
            "adam": optimizers.Adam
        }

    @abstractmethod
    def loss(self, output: tuple | Any):
        raise NotImplementedError()

    @abstractmethod
    def optimizer(self):
        raise NotImplementedError()

class Generator(BaseModel):
    def __init__(self, unit, noise_dimension, frame_size, kernel, opt, lr):
        super().__init__()
        self.noise_dimension = noise_dimension
        self.frame_size = frame_size
        self.unit = unit
        self.norm = layers.BatchNormalization()
        self.activation = layers.LeakyReLU()
        self.reshaping = layers.Reshape
        self.kernel = kernel
        self.opt = opt
        self.lr = lr

    def loss(self, fake_output):
        return self.cross_entropy_B(
            tf.ones_like(fake_output),
            fake_output
        )

    def optimizer(self):
        return self.optimizers.get(self.opt)(self.lr)

    def dense_layer(self, multiplier):
        return layers.Dense(
            (
                multiplier
                *multiplier
                *self.unit
            ),
            use_bias=False
        )

    def reshaping(self, multiplier):
        return layers.Reshape(
            (
                multiplier,
                multiplier,
                self.unit
            )
        )

    def conv2_transpose(self, unit, stride, activation=None) -> layers.Conv2DTranspose:
        _layer = layers.Conv2DTranspose(
            unit,
            (self.kernel, self.kernel),
            strides=(stride, stride),
            padding='same',
            use_bias=False
        )
        if activation is not None:
            _layer.activation = activation
        return _layer

    def call(self, _input: tf.Tensor) -> tf.Tensor:
        _dense = self.dense_layer(int(self.frame_size[0]/8))(_input)
        _dense = self.norm(_dense)
        _dense = self.activation(_dense)
        _dense = self.reshaping(int(self.frame_size[0]/8))(_dense)
        _dense = self.conv2_transpose(int(self.unit/2), 1)(_dense)
        _dense = self.norm(_dense)
        _dense = self.activation(_dense)

        _dense = self.conv2_transpose(int(self.unit/4), 2)(_dense)
        _dense = self.norm(_dense)
        _dense = self.activation(_dense)

        _dense = self.conv2_transpose(int(self.unit/8), 2)(_dense)
        _dense = self.norm(_dense)
        _dense = self.activation(_dense)

        _dense = self.conv2_transpose(3, 2, "tanh")(_dense)
        return _dense

class Discriminator(BaseModel):
    def __init__(self, unit, noise_dimension, frame_size, kernel, opt, lr, dropout_ratio):
        super().__init__()
        self.noise_dimension = noise_dimension
        self.frame_size = frame_size
        self.unit = unit
        self.activation = layers.LeakyReLU()
        self.dropout = layers.Dropout(dropout_ratio)
        self.dense_layer = layers.Dense
        self.flatten = layers.Flatten()
        self.kernel = kernel
        self.opt = opt
        self.lr = lr

    def loss(self, output):
        real_output, fake_output = output
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def optimizer(self):
        return self.optimizers.get(self.opt)(self.lr)

    def conv2(self, unit, stride, activation=None) -> layers.Conv2D:
        _layer = layers.Conv2D(
            unit,
            (self.kernel, self.kernel),
            strides=(stride, stride),
            padding='same',
            use_bias=False
        )
        if activation is not None:
            _layer.activation = activation
        return _layer


    def call(self, _input: tf.Tensor):
        _dense = self.conv2(self.unit, 2)(_input)
        _dense = self.activation(_dense)
        _dense = self.dropout(_dense)
        _dense = self.conv2(int(self.unit*2), 2)(_dense)
        _dense = self.activation(_dense)
        _dense = self.dropout(_dense)

        _dense = self.conv2(int(self.unit*4), 2)(_dense)
        _dense = self.activation(_dense)
        _dense = self.dropout(_dense)

        _dense = self.conv2(int(self.unit*8), 2)(_dense)
        _dense = self.activation(_dense)
        _dense = self.dropout(_dense)
        _dense = self.flatten(_dense)
        return self.dense_layer(_dense)


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


    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.summary()
    return model
