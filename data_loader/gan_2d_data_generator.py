import math
import numpy as np
from skimage import transform
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils


class Gan2dDataGenerator(BaseDataGenerator):
    def __init__(self, dataset, batch_size, image_size, channels, is_training, **kwargs):
        BaseDataGenerator.__init__(self, dataset, batch_size, image_size, channels, is_training)

    def get_size(self):
        size = 0
        for item in self.dataset:
            npz = np.load(item)
            size += npz['A'].shape[2]
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
                for s_id in range(a_nii.shape[2]):
                    a, b = self.get_multi_channel_image(s_id, a_nii, b_nii)
                    batchA.append(a)
                    batchB.append(b)
                    if len(batchA) == self.batch_size:
                        yield a_path, np.array(batchA), b_path, np.array(batchB)
                        batchA = list()
                        batchB = list()
            for _ in range(self.batch_size - len(batchA)):
                batchA.append(np.zeros((self.image_size[0], self.image_size[1], self.channels), dtype=float))
                batchB.append(np.zeros((self.image_size[0], self.image_size[1], self.channels), dtype=float))
            yield a_path, np.array(batchA), b_path, np.array(batchB)

    def get_multi_channel_image(self, s_id, a_nii, b_nii):
        channels_imagesA = []
        channels_imagesB = []
        for _ in range(s_id, self.channels // 2):
            channels_imagesA.append(np.zeros(self.image_size, dtype=float))
            channels_imagesB.append(np.zeros(self.image_size, dtype=float))
        padding = len(channels_imagesA)

        for _id in range(s_id - self.channels // 2 + padding,
                         min(s_id + self.channels - self.channels // 2, a_nii.shape[2])):
            sliceA = transform.resize(a_nii[:, :, _id], self.image_size)
            sliceB = transform.resize(b_nii[:, :, _id], self.image_size)
            if self.is_training:
                # todo 数据增广
                pass
            channels_imagesA.append(sliceA)
            channels_imagesB.append(sliceB)
        padding = len(channels_imagesA)

        for _ in range(self.channels - padding):
            channels_imagesA.append(np.zeros(self.image_size, dtype=float))
            channels_imagesB.append(np.zeros(self.image_size, dtype=float))

        channels_imagesA = np.array(channels_imagesA).transpose((1, 2, 0))
        channels_imagesB = np.array(channels_imagesB).transpose((1, 2, 0))
        return channels_imagesA, channels_imagesB


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Datasets/Neurofibromatosis/t12stir_train.yaml')
    train_generator = Gan2dDataGenerator(dataset, 8, (512, 256), 1, True)
    data = train_generator.get_data_generator()
    print(next(data))
