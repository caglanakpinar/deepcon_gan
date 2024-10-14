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


@dataclass
class Params(Paths):
    batch_size: int = 32
    epochs: int = 5
    lr: float | list[float] = 1e-4

    @staticmethod
    def default_parameters():
        return Params.__dict__.get('__annotations__').keys()

    @classmethod
    def get(cls, p):
        assert getattr(cls, p, None) is not None, f"{p} - is not available at train parameters .yaml file"
        return getattr(cls, p)

    @classmethod
    def set(cls, p, value):
        setattr(cls, p, value)
        return cls

    @classmethod
    def read_from_config(cls, trainer_config_path, **kwargs):
        _params = cls.read_yaml(cls.parent_dir / trainer_config_path)
        parameters = Params()
        for p, value in _params.items():
            parameters = cls.set(p, value)
        if kwargs is not None:
            for p, value in kwargs.items():
                parameters = cls.set(p, value)
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


class CapturingImages(Paths):
    def __init__(self, params: Params, images=None):
        self.batch_size = params.get("batch_size")
        self.image_size = params.get("image_size")
        self.buffer_size = params.get("buffer_size")
        self.images = images
        self.image_directory = self.create_image_directory(params.get("name"))

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
        return (images - 127.5) / 127.5

    @classmethod
    def read_images(
            cls,
            params: Params
    ):
        ci = CapturingImages(params)
        if not (ci.parent_dir / params.get("name")).exists():
            ci.capture_images(name=params.get("name"), buffer_size=params.get("buffer_size"), image_size=params.get("image_size"))
        data = np.array(
            [
                ci.normalize_image(
                    image.resize(
                        ci.read_image(file_name),
                        [params.get("image_size")[0], params.get("image_size")[1]],
                        method=image.ResizeMethod.NEAREST_NEIGHBOR
                    ).numpy().astype('float32')
                )
                for file_name in ci.image_directory.iterdir()
            ]
        )
        ci.images = data.reshape(data.shape[0], params.get("image_size")[0], params.get("image_size")[1], params.get("image_size")[2]).astype('float32')
        ci.create_tfds()
        return ci
