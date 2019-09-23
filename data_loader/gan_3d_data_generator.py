import math
import numpy as np
from skimage import transform
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils


class Gan3dDataGenerator(BaseDataGenerator):
    def __init__(self, dataset, batch_size, image_size, channels, is_training, **kwargs):
        BaseDataGenerator.__init__(self, dataset, batch_size, image_size, channels, is_training)

    def get_size(self):
        size = 0
        for item in self.dataset:
            npz = np.load(item)
            size += npz['A'].shape[0]
        return math.ceil(size / self.batch_size)

    def get_data_generator(self):
        while True:
            batchA = list()
            batchB = list()
            a_path = ''
            b_path = ''
            for item in self.dataset:
                npz = np.load(item)
                a_nii = npz['A']
                a_path = npz['A_path']
                b_nii = npz['B']
                b_path = npz['B_path']
                a, b = self.get_multi_channel_image(a_nii, b_nii)
                batchA.append(a)
                batchB.append(b)
                if len(batchA) == self.batch_size:
                    yield a_path, np.array(batchA), b_path, np.array(batchB)
                    batchA = list()
                    batchB = list()
            for _ in range(self.batch_size - len(batchA)):
                batchA.append(
                    np.zeros((self.image_size[0], self.image_size[1], self.image_size[2], self.channels), dtype=float))
                batchB.append(
                    np.zeros((self.image_size[0], self.image_size[1], self.image_size[2], self.channels), dtype=float))
            yield a_path, np.array(batchA), b_path, np.array(batchB)

    def get_multi_channel_image(self, a_nii, b_nii):
        channels_imagesA = list()
        channels_imagesB = list()
        a_nii = transform.resize(a_nii, self.image_size)
        b_nii = transform.resize(b_nii, self.image_size)
        if self.is_training:
            # todo 数据增广
            pass
        channels_imagesA.append(a_nii)
        channels_imagesB.append(b_nii)

        channels_imagesA = np.array(channels_imagesA).transpose((1, 2, 3, 0))
        channels_imagesB = np.array(channels_imagesB).transpose((1, 2, 3, 0))
        return channels_imagesA, channels_imagesB


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Datasets/Neurofibromatosis/t12stir_train.yaml')
    train_generator = Gan3dDataGenerator(dataset, 8, (512, 256), 1, True)
    data = train_generator.get_data_generator()
    print(next(data))
