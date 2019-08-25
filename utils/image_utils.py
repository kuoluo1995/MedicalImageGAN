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

