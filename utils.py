from dataclasses import dataclass

import yaml
import PIL
import cv2
from pathlib import Path
import numpy as np
from tensorflow import image, data


class Paths:
    checkpoint_dir = 'training_checkpoints'
    parent_dir = Path(__file__).absolute().parent

    def create_image_directory(self, name):
        file_path = self.parent_dir / Path(name)
        if not file_path.exists():
            Path.mkdir(self.parent_dir/ Path(name))
        return file_path

    @staticmethod
    def checkpoint_directory(name) -> Path:
        return Paths.parent_dir / f"{Paths.checkpoint_dir}_{name.upper()}"

    def create_train_epoch_image_save(self, name):
        folder_path = self.checkpoint_directory(name) / "ckpt"
        if not folder_path.exists():
            Path.mkdir(self.checkpoint_directory(name))
            Path.mkdir(folder_path)
        return folder_path


class CapturingImages(Paths):
    def __init__(self, name, batch_size, image_size, buffer_size, images=None):
        self.batch_size = batch_size
        self.image_size = image_size
        self.buffer_size = buffer_size
        self.images = images
        self.image_directory = self.create_image_directory(name)

    def create_tfds(self):
        self.images = (
            data.Dataset
            .from_tensor_slices(self.images)
            .shuffle(self.images.shape[0]).batch(self.batch_size)
        )

    def capture_images(self, name, buffer_size, image_size: tuple):
        vidcap = cv2.VideoCapture(0)
        if not vidcap.isOpened():
            print("Cannot open camera")
            exit()
        count = 0
        dir = self.create_image_directory(name)
        success, image = vidcap.read()
        while success:
            try:
                success, image = vidcap.read()
                image = cv2.resize(image, (image_size[0], image_size[0]))
                file_name = "frame%d.jpg" % count
                cv2.imwrite(dir / file_name, image)  # save frame as JPEG file
                count += 1
                if count >= buffer_size:
                    success = False
            except:
                pass

    @staticmethod
    def read_image(file_name):
        img = PIL.Image.open(file_name)
        img.load()
        return np.asarray(img, dtype="int32")

    @staticmethod
    def normalize_image(images):
        return  (images - 127.5) / 127.5

    @classmethod
    def read_images(cls, name, batch_size, image_size, buffer_size):
        ci = CapturingImages(name, batch_size, image_size, buffer_size)
        if not (ci.parent_dir / name).exists():
            ci.capture_images(name=name, buffer_size=buffer_size, image_size=image_size)
        data = np.array(
            [
                ci.normalize_image(
                    image.resize(
                        ci.read_image(file_name),
                        [image_size[0], image_size[1]],
                        method=image.ResizeMethod.NEAREST_NEIGHBOR
                    ).numpy().astype('float32')
                )
                for file_name in ci.image_directory.iterdir()
            ]
        )
        ci.images = data.reshape(data.shape[0], image_size[0], image_size[1], image_size[2]).astype('float32')
        ci.create_tfds()
        return ci

@dataclass
class BaseParams:
    batch_size: int
    epochs: int
    lr: float | list[float] = 1e-4

    @staticmethod
    def default_parameters():
        return BaseParams.__dict__.get('__annotations__').keys()



class Params(BaseParams, Paths):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def read_from_config(cls, config_name, config_path=""):
        _params = cls.read_yaml(cls.parent_dir / config_path / config_name)
        parameters = Params(**{p: value for p, value in  _params.items() if p in cls.default_parameters()})
        for p, value in _params.items():
            if p  not in cls.default_parameters():
                setattr(parameters, p, value)
        return parameters

    @staticmethod
    def read_yaml(folder):
        """
        :param folder: file path ending with .yaml format
        :return: dictionary
        """
        with open(f"{str(folder)}.yaml" if str(folder).split(".")[-1] not in ['yaml', 'yml'] else folder) as file:
            docs = yaml.full_load(file)
        return docs

