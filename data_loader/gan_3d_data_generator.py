import math
import numpy as np
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils
from utils.nii_utils import nii_header_reader, nii_writer


class Gan3dDataGenerator(BaseDataGenerator):
    def __init__(self, dataset_list, batch_size, image_size, channels, is_training, base_patch=32, **kwargs):
        BaseDataGenerator.__init__(self, dataset_list, batch_size, image_size, channels, is_training, base_patch)

    def get_size(self):
        size = len(self.dataset_list)
        shape = np.array(self.image_size) // self.base_patch
        size *= shape[0] * shape[1] * shape[2]
        return math.ceil(size / self.batch_size)

    def get_num_patch(self):
        shape = self.image_size // self.base_patch
        return shape[0] * shape[1] * shape[2]

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
                a_path = npz['A_path']
                b_path = npz['B_path']
                shape = np.array(self.image_size) // self.base_patch
                for n_d in range(shape[0]):
                    for n_h in range(shape[1]):
                        for n_w in range(shape[2]):
                            a_patch = a_nii[(n_d - 1) * self.base_patch:self.base_patch * n_d,
                                      self.base_patch * (n_h - 1):self.base_patch * n_h,
                                      self.base_patch * (n_w - 1):self.base_patch * n_w]
                            b_patch = b_nii[(n_d - 1) * self.base_patch:self.base_patch * n_d,
                                      self.base_patch * (n_h - 1):self.base_patch * n_h,
                                      self.base_patch * (n_w - 1):self.base_patch * n_w]
                            a_patch = self.get_multi_channel_image(a_patch)
                            b_patch = self.get_multi_channel_image(b_patch)
                            batchA.append(a_patch)
                            batchB.append(b_patch)
                            if len(batchA) == self.batch_size:
                                yield a_path, np.array(batchA), b_path, np.array(batchB)
                                batchA = list()
                                batchB = list()
                for _ in range(self.batch_size - len(batchA)):  # one batch(all patch in the same images)
                    batchA.append(
                        np.zeros((self.base_patch, self.base_patch, self.base_patch, self.channels), dtype=float))
                    batchB.append(
                        np.zeros((self.base_patch, self.base_patch, self.base_patch, self.channels), dtype=float))
                yield a_path, np.array(batchA), b_path, np.array(batchB)

    def get_multi_channel_image(self, data):
        channels_images = list()
        if self.is_training:
            # todo 数据增广
            pass
        channels_images.append(data)
        channels_images = np.array(channels_images).transpose((1, 2, 3, 0))
        return channels_images


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Dataset/Neurofibromatosis/T12STIR_train.yaml')
    data_loader = Gan3dDataGenerator(dataset, 1, 1, True)
    generator = data_loader.get_data_generator()
    a_path, batchA, b_path, batchB = next(generator)
    b_nii_head = nii_header_reader(b_path)
    nii_writer('./fake.nii', b_nii_head, np.squeeze(batchB))
