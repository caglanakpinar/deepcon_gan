from abc import abstractmethod

import keras_tuner
import keras
from keras_tuner import HyperParameters
from mlp import Params, BaseHyperModel


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
        _model = self.temp_model(self.search_params)
        _model.train(train_dataset=kwargs['x'], val_dataset=kwargs['validation_data'], tunable=True)
        return {
            'loss': _model.get_best_epoch_loss()
        }

