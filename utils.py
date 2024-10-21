import PIL
import cv2
import numpy as np
from tensorflow import image, data

from mlp import Paths, Params, BaseData, log


class CapturingImages(BaseData, Paths):
    def __init__(self, params: Params):
        self.params = params
        self.batch_size = params.get("batch_size")
        self.image_size = params.get("image_size")
        self.buffer_size = params.get("buffer_size")
        self.image_directory = self.create_directory_in_parents(params.get("name"))

    def create_tfds(self):
        self.data = (
            data.Dataset
            .from_tensor_slices(self.data)
            .shuffle(self.data.shape[0]).batch(self.batch_size)
        )

    def capture_images(self, name, buffer_size, image_size: tuple):
        vidcap = cv2.VideoCapture(0)
        if not vidcap.isOpened():
            log(log.error, "Cannot open camera")
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
    def read(
            cls,
            params: Params,
            **kwargs
    ):
        ci = CapturingImages(params)
        if len([f for f in ci.image_directory.iterdir()]) == 0:  # 3 no image in directory
            ci.capture_images(
                name=params.get("name"),
                buffer_size=params.get("buffer_size"),
                image_size=params.get("image_size")
            )
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
        ci.data = data.reshape(
            data.shape[0],
            params.get("image_size")[0],
            params.get("image_size")[1],
            params.get("image_size")[2]
        ).astype('float32')
        ci.create_tfds()
        return ci
