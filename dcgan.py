import keras
import tensorflow as tf
from keras import *
import time
from IPython import display
import matplotlib.pyplot as plt
from keras_tuner import HyperParameters, RandomSearch

from utils import Paths, Params
from models import Generator, cross_entropy, Discriminator
from mlp import BaseModel, BaseHyperModel, log


class DCGAN(BaseModel, Paths):
    def __init__(
        self,
        params: Params
    ):
        self.params: Params = params
        self.name = params.get('name')
        self.generator = Generator.make_model(params)
        self.discriminator = Discriminator.make_model(params)
        self.generator_optimizer = optimizers.Adam(params.get('lr_generator'))
        self.discriminator_optimizer = optimizers.Adam(params.get('lr_discriminator'))
        self.generator_loss = Generator.loss
        self.discriminator_loss = Discriminator.loss
        self.noise_dim = params.get('noise_dimension')
        self.batch_size = params.get('batch_size')
        self.seed = tf.random.normal([1, self.noise_dim])
        self.epochs = params.get('epochs')
        self.checkpoint_save_epoch = params.get('checkpoint_save_epoch')
        self.epoch_loss_metric = keras.metrics.Sum()

    @classmethod
    def read_checkpoint(cls, params: Params):
        checkpoint_path = cls.checkpoint_directory(params.get('name'))
        if not checkpoint_path.exists():
            raise f"{checkpoint_path} - not found"
        dcgan = DCGAN(params)
        latest = tf.train.latest_checkpoint(checkpoint_path)
        dcgan.checkpoint.restore(
            latest
        )
        return dcgan

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

    def train(self, dataset: tf.data.Dataset, val_dataset: tf.data.Dataset = None, tunable=False):
        for epoch in range(self.epochs):
            start = time.time()
            for image_batch in dataset:
                self.train_step(image_batch)
            if tunable:
                for images in val_dataset:
                    generated_image = self.generator(tf.random.normal([images.shape[0], self.noise_dim]), training=False)
                    loss = cross_entropy(images, generated_image)
                    self.epoch_loss_metric.update_state(loss)
            else:
                # Produce images for the GIF as you go
                display.clear_output(wait=True)
                # Save the model every x epochs - it is in params.yaml
                if (epoch + 1) % self.checkpoint_save_epoch == 0:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix_directory(self.name))
                log(log.info, 'Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
                self.generate_and_save_images(
                    epoch + 1,
                    self.seed,
                    show=False
                )

    def get_best_epoch_loss(self):
        return float(self.epoch_loss_metric.result().numpy())

    def generate_and_save_images(self, epoch, test_input, show=False):
        predictions = self.generator(test_input, training=False)
        plt.imshow(((predictions[0].numpy() * 127.5) + 127.5).astype(int))
        plt.axis('off')

        plt.savefig(self.checkpoint_prefix_directory(self.name) / 'image_at_epoch_{:04d}.png'.format(epoch))
        if show:
            plt.show()


class HyperDCGAN(BaseHyperModel):
    def build(self, hp: HyperParameters):
        _selection_args = {
            p: (
                hp.Choice(p, getattr(self.temp_hyper_params, p))
                if type(getattr(self.temp_hyper_params, p)) == list
                else getattr(self.temp_hyper_params, p)
            )
             for p in self.temp_hyper_params.parameter_keys
        }
        _args = {
            p: (_selection_args.get(p) if p in _selection_args.keys() else getattr(self.temp_train_args, p))
            for p in self.temp_train_args.parameter_keys
        }
        self.search_params = Params(trainer_arguments=_args)
        return keras.Model()

    def fit(self, fp, model: keras.Model, **kwargs):
        _model = self.temp_model(
            self.search_params
        )
        _model.train(dataset=kwargs['x'], val_dataset=kwargs['x'], tunable=True)
        return {
            'loss': _model.get_best_epoch_loss()
        }
