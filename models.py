from keras import layers
import keras
import tensorflow as tf

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = keras.optimizers.Adam(0.01)
discriminator_optimizer = keras.optimizers.Adam(0.01)

def generator_loss(real_output, fake_output):
    return cross_entropy(real_output, fake_output)


def discriminator_loss(real_output, fake_output):
    return cross_entropy(fake_output, real_output)


def make_generator_model(unit, batch_size, frame_size, noise_dimension) -> keras.Sequential:
    model = keras.Sequential()
    model.add(layers.Conv2DTranspose(filters=unit, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False,
                                     input_shape=[noise_dimension, noise_dimension, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=int(unit/2), kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False,
                                     input_shape=[noise_dimension, noise_dimension, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(filters=int(unit/4), kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False,
                                     input_shape=[noise_dimension, noise_dimension, 3]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, frame_size[0], frame_size[1], 3)
    model.summary()
    return model


def make_discriminator_model(unit, frame_size) -> keras.Sequential:
    model = keras.Sequential()
    model.add(layers.Conv2D(3, (3, 3), strides=(1, 1), padding='same',
                                     input_shape=[frame_size[0], frame_size[1], 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.summary()
    return model
