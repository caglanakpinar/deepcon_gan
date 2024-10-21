import ssl

from mlp.cli.cli import cli

ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == "__main__":
    cli()


# train model
"""
poetry run python main.py \
model train \
--training_class dcgan.DCGAN \
--trainer_config_path configs/params.yaml \
--data_access_clas utils.CapturingImages \
--continuous_training True \
"""
# hyperparameter tuning
# train model
"""
poetry run python main.py \
model train \
--tuning_class dcgan.HyperDCGAN \
--training_class dcgan.DCGAN \
--trainer_config_path configs/params.yaml \
--hyperparameter_config_path configs/hyper_params.yaml \
--data_access_clas utils.CapturingImages \
--continuous_training True
"""
