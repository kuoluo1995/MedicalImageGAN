import copy
import numpy as np


class ImagePool(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.imagesA = []
        self.imagesB = []

    def __call__(self, imageA, imageB):
        if self.maxsize <= 0:
            return imageA, imageB
        if len(self.imagesA) < self.maxsize:
            self.imagesA.append(imageA)
            self.imagesB.append(imageB)
            return imageA, imageB
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            _imageA = copy.copy(self.imagesA[idx])
            self.imagesA[idx] = imageA
            idx = int(np.random.rand() * self.maxsize)
            _imageB = copy.copy(self.imagesB[idx])
            self.imagesB[idx] = imageB
            return _imageA, _imageB
        else:
            return imageA, imageB
