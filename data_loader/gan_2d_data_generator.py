import math
import numpy as np
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
            for item in self.dataset_list:
                npz = np.load(item)
                a_nii = npz['A']
                b_nii = npz['B']
                a_path = str(npz['A_path'])
                b_path = str(npz['B_path'])
                for s_id in range(a_nii.shape[2]):
                    a_image, b_image = self.get_multi_channel_image(s_id, a_nii, self.in_channels, b_nii,
                                                                    self.out_channels)
                    batchA.append(a_image)
                    batchB.append(b_image)
                    if len(batchA) == self.batch_size:
                        yield a_path, np.array(batchA), b_path, np.array(batchB)
                        batchA = list()
                        batchB = list()
                if len(batchA) > 0:
                    for _ in range(self.batch_size - len(batchA)):
                        batchA.append(np.zeros((self.image_size[0], self.image_size[1], self.in_channels), dtype=float))
                        batchB.append(
                            np.zeros((self.image_size[0], self.image_size[1], self.out_channels), dtype=float))
                    yield a_path, np.array(batchA), b_path, np.array(batchB)
                    batchA = list()
                    batchB = list()

    def get_multi_channel_image(self, s_id, a_data, a_channels, b_data, b_channels):
        itensity = 0
        if self.is_training:  # todo 数据增广
            itensity = np.random.rand() * 0.1

        channels_images_a = []
        for _ in range(s_id, a_channels // 2):
            channels_images_a.append(np.zeros((self.image_size[0], self.image_size[1]), dtype=float))
        padding = len(channels_images_a)
        for _id in range(s_id - a_channels // 2 + padding, min(s_id + a_channels - a_channels // 2, a_data.shape[2])):
            channels_images_a.append(a_data[:, :, _id] + itensity)
        padding = len(channels_images_a)
        for _ in range(a_channels - padding):
            channels_images_a.append(np.zeros((self.image_size[0], self.image_size[1]), dtype=float))
        channels_images_a = np.array(channels_images_a).transpose((1, 2, 0))

        channels_images_b = []
        for _ in range(s_id, b_channels // 2):
            channels_images_b.append(np.zeros((self.image_size[0], self.image_size[1]), dtype=float))
        padding = len(channels_images_b)
        for _id in range(s_id - b_channels // 2 + padding, min(s_id + b_channels - b_channels // 2, b_data.shape[2])):
            channels_images_b.append(b_data[:, :, _id] + itensity)
        padding = len(channels_images_b)
        for _ in range(b_channels - padding):
            channels_images_b.append(np.zeros((self.image_size[0], self.image_size[1]), dtype=float))
        channels_images_b = np.array(channels_images_b).transpose((1, 2, 0))
        return channels_images_a, channels_images_b


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Dataset/Neurofibromatosis/t12stir_train.yaml')
    train_generator = Gan2dDataGenerator(dataset, 8, (512, 256), 1, True)
    data = train_generator.get_data_generator()
    print(next(data))
