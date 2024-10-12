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
