import math
import numpy as np
from skimage import transform
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils


class Gan2dDataGenerator(BaseDataGenerator):
    def __init__(self, dataset_list, batch_size, image_size, in_channels, out_channels, is_training, **kwargs):
        BaseDataGenerator.__init__(self, dataset_list, batch_size, image_size, in_channels, out_channels, is_training)

    def get_size(self):
        size = 0
        for item in self.dataset_list:
            npz = np.load(item)
            size += npz['A'].shape[2]
        return math.ceil(size / self.batch_size)

    def get_image_size(self):
        return self.image_size

    def get_data_generator(self):
        while True:
            batchA = list()
            batchB = list()
            a_path = ''
            b_path = ''
            for item in self.dataset_list:
                npz = np.load(item)
                a_nii = npz['A']
                b_nii = npz['B']
                a_path = npz['A_path']
                b_path = npz['B_path']
                for s_id in range(a_nii.shape[2]):
                    a_image = self.get_multi_channel_image(s_id, a_nii, self.in_channels)
                    b_image = self.get_multi_channel_image(s_id, b_nii, self.out_channels)
                    batchA.append(a_image)
                    batchB.append(b_image)
                    if len(batchA) == self.batch_size:
                        yield a_path, np.array(batchA), b_path, np.array(batchB)
                        batchA = list()
                        batchB = list()
            for _ in range(self.batch_size - len(batchA)):
                batchA.append(np.zeros((self.image_size[0], self.image_size[1], self.channels), dtype=float))
                batchB.append(np.zeros((self.image_size[0], self.image_size[1], self.channels), dtype=float))
            yield a_path, np.array(batchA), b_path, np.array(batchB)

    def get_multi_channel_image(self, s_id, data, channels):
        channels_images = []
        for _ in range(s_id, channels // 2):
            channels_images.append(np.zeros(self.image_size, dtype=float))
        padding = len(channels_images)

        for _id in range(s_id - channels // 2 + padding, min(s_id + channels - channels // 2, data.shape[2])):
            if self.is_training:
                # todo 数据增广
                pass
            channels_images.append(data[:, :, _id])
        padding = len(channels_images)

        for _ in range(channels - padding):
            channels_images.append(np.zeros(self.image_size, dtype=float))

        channels_images = np.array(channels_images).transpose((1, 2, 0))
        return channels_images


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Datasets/Neurofibromatosis/t12stir_train.yaml')
    train_generator = Gan2dDataGenerator(dataset, 8, (512, 256), 1, True)
    data = train_generator.get_data_generator()
    print(next(data))
