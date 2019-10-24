import math
import numpy as np
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils


class Gan2dDataGenerator(BaseDataGenerator):
    def __init__(self, is_training, dataset_list, data_shape, batch_size, in_channels, out_channels, **kwargs):
        BaseDataGenerator.__init__(self, is_training, dataset_list, data_shape, batch_size, in_channels, out_channels)

    def get_size(self):
        size = 0
        for item in self.dataset_list:
            npz = np.load(item)
            size += npz['A'].shape[2]
        return math.ceil(size / self.batch_size)

    def get_data_shape(self):
        return self.data_shape

    def get_data_generator(self):
        while True:
            batch_a = list()
            batch_b = list()
            for item in self.dataset_list:
                npz = np.load(item)
                nii_a = npz['A']
                nii_b = npz['B']
                path_a = str(npz['path_a'])
                path_b = str(npz['path_b'])
                source_path_a = str(npz['source_path_a'])
                source_path_b = str(npz['source_path_b'])
                for s_id in range(nii_a.shape[2]):
                    image_a, image_b = self.get_multi_channel_image(s_id, nii_a, self.in_channels, nii_b,
                                                                    self.out_channels)
                    batch_a.append(image_a)
                    batch_b.append(image_b)
                    if len(batch_a) == self.batch_size:
                        yield path_a, source_path_a, np.array(batch_a), path_b, source_path_b, np.array(batch_b)
                        batch_a = list()
                        batch_b = list()
                if len(batch_a) > 0:
                    for _ in range(self.batch_size - len(batch_a)):
                        batch_a.append(
                            np.zeros((self.data_shape[0], self.data_shape[1], self.in_channels), dtype=float))
                        batch_b.append(
                            np.zeros((self.data_shape[0], self.data_shape[1], self.out_channels), dtype=float))
                    yield path_a, source_path_a, np.array(batch_a), path_b, source_path_b, np.array(batch_b)
                    batch_a = list()
                    batch_b = list()

    def get_multi_channel_image(self, s_id, nii_a, channels_a, nii_b, channels_b):
        channels_images_a = []
        channels_images_b = []
        for _ in range(s_id, channels_a // 2):
            channels_images_a.append(np.zeros((self.data_shape[0], self.data_shape[1]), dtype=float))
        for _ in range(s_id, channels_b // 2):
            channels_images_b.append(np.zeros((self.data_shape[0], self.data_shape[1]), dtype=float))
        padding_a = len(channels_images_a)
        padding_b = len(channels_images_b)

        itensity = 0
        if self.is_training:  # todo 数据增广
            itensity = np.random.rand() * 0.1
        for _id in range(s_id - channels_a // 2 + padding_a, min(s_id + channels_a - channels_a // 2, nii_a.shape[2])):
            channels_images_a.append(nii_a[:, :, _id] + itensity)
        for _id in range(s_id - channels_b // 2 + padding_b, min(s_id + channels_b - channels_b // 2, nii_b.shape[2])):
            channels_images_b.append(nii_b[:, :, _id] + itensity)
        padding_a = len(channels_images_a)
        padding_b = len(channels_images_b)

        for _ in range(channels_a - padding_a):
            channels_images_a.append(np.zeros((self.data_shape[0], self.data_shape[1]), dtype=float))
        for _ in range(channels_b - padding_b):
            channels_images_b.append(np.zeros((self.data_shape[0], self.data_shape[1]), dtype=float))
        channels_images_a = np.array(channels_images_a).transpose((1, 2, 0))
        channels_images_b = np.array(channels_images_b).transpose((1, 2, 0))
        return channels_images_a, channels_images_b


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Dataset/Neurofibromatosis/t12stir_train.yaml')
    train_generator = Gan2dDataGenerator(dataset, 8, (512, 256), 1, True)
    data = train_generator.get_data_generator()
    print(next(data))
