import numpy as np
import copy
from scipy import misc
from utils import nii_utils


class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.images = []

    def __call__(self, image):
        if self.maxsize <= 0:
            return image
        if len(self.images) < self.maxsize:
            self.images.append(image)
            return image
        if np.random.rand() > 0.5:
            id = int(np.random.rand() * self.maxsize)
            A = copy.copy(self.images[id])[0]
            self.images[id][0] = image[0]
            # id = int(np.random.rand() * self.maxsize)
            B = copy.copy(self.images[id])[1]
            self.images[id][1] = image[1]
            return [A, B]
        else:
            return image


def save_images(images, size, image_path):
    return misc.imsave(images, size, image_path)


def load_data(images_path, image_size, batch_size, is_training=True):  # todo batch_size
    result = list()
    image_A = nii_utils.nii_reader(images_path[0])
    image_A = np.transpose(image_A, (2, 0, 1))
    image_A = np.clip(image_A, 0, 1500) / 1500
    image_B = nii_utils.nii_reader(images_path[1])
    image_B = np.transpose(image_B, (2, 0, 1))
    image_B = np.clip(image_B, 0, 380) / 380
    for i in range(image_A.shape[0]):
        a = np.resize(image_A[i], (image_size[0], image_size[1]))[np.newaxis, :, :, np.newaxis]
        b = np.resize(image_B[i], (image_size[0], image_size[1]))[np.newaxis, :, :, np.newaxis]
        result.append(np.concatenate([a, b], axis=3))
    return result
