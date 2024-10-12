import PIL
import cv2
from pathlib import Path
import numpy as np
from tensorflow import image, data


class Paths:
    checkpoint_dir = './training_checkpoints'

    @staticmethod
    def create_image_directory(name):
        file_path = Path(__file__).absolute().parent / Path(name)
        if not file_path.exists():
            Path.mkdir(Path(__file__).absolute().parent / Path(name))
        return file_path

    def create_train_checkpoint_directory(self, name):
        folder_path = Path(__file__).absolute().parent / f"training_checkpoints_{name.upper()}" / "ckpt" / name
        if not folder_path.exists():
            Path.mkdir(Path(__file__).absolute().parent / f"training_checkpoints_{name.upper()}")
            Path.mkdir(Path(__file__).absolute().parent / f"training_checkpoints_{name.upper()}" / "ckpt")
            Path.mkdir(folder_path)
        return folder_path


class CapturingImages(Paths):
    def __init__(self, batch_size=128, images=None):
        self.batch_size = batch_size
        self.images = images

    def create_tfds(self):
        self.images = (
            data.Dataset
            .from_tensor_slices(self.images)
            .shuffle(self.images.shape[0]).batch(self.batch_size)
        )

    def capture_images(self, name, buffer_size, frame_size: tuple):
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
                image = cv2.resize(image, (frame_size[0], frame_size[0]))
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
    def read_images(cls, name, frame_size):
        dir = cls.create_image_directory(name)
        data = np.array(
            [
                cls.normalize_image(
                    image.resize(
                        cls.read_image(file_name),
                        [frame_size[0], frame_size[1]],
                        method=image.ResizeMethod.NEAREST_NEIGHBOR
                    ).numpy().astype('float32')
                )
                for file_name in dir.iterdir()
            ]
        )
        return cls(
            images=data.reshape(data.shape[0], frame_size[0], frame_size[1], 3).astype('float32')
        )
