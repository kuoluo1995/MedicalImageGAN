import cv2
import math
import numpy as np

from data_loader import preprocess
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils


class DataGeneratorGan2d(BaseDataGenerator):
    def __init__(self, is_augmented, dataset_list, data_shape, batch_size, in_channels, out_channels, **kwargs):
        BaseDataGenerator.__init__(self, is_augmented, dataset_list, data_shape, batch_size, in_channels, out_channels)

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
            if self.is_augmented:
                np.random.shuffle(self.dataset_list)
            batch_a = list()
            batch_b = list()
            for item in self.dataset_list:
                npz = np.load(item)
                nii_a = preprocess(npz['A'])  # [0, 1] => [-1, 1]
                nii_b = preprocess(npz['B'])  # [0, 1] => [-1, 1]
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

        flip = False
        offset_x, offset_y = (0, 0)
        if self.is_augmented:
            flip = True if np.random.rand() > 0.5 else False
            scale_size = (286, 286)
            offset_y = np.random.randint(0, scale_size[0] - self.data_shape[0])
            offset_x = np.random.randint(0, scale_size[1] - self.data_shape[1])
        else:
            scale_size = self.data_shape
        for _id in range(s_id - channels_a // 2 + padding_a, min(s_id + channels_a - channels_a // 2, nii_a.shape[2])):
            if self.is_augmented:
                slice_a = cv2.resize(nii_a[:, :, _id], scale_size, interpolation=cv2.INTER_AREA)
                slice_a = slice_a[offset_y:offset_y + self.data_shape[0], offset_x:offset_x + self.data_shape[1]]
                if flip:
                    slice_a = np.fliplr(slice_a)
            else:
                slice_a = nii_a[:, :, _id]
            channels_images_a.append(slice_a)
        for _id in range(s_id - channels_b // 2 + padding_b, min(s_id + channels_b - channels_b // 2, nii_b.shape[2])):
            if self.is_augmented:
                slice_b = cv2.resize(nii_b[:, :, _id], scale_size, interpolation=cv2.INTER_AREA)
                slice_b = slice_b[offset_y:offset_y + self.data_shape[0], offset_x:offset_x + self.data_shape[1]]
                if flip:
                    slice_b = np.fliplr(slice_b)
            else:
                slice_b = nii_b[:, :, _id]
            channels_images_b.append(slice_b)
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
    dataset_dict = yaml_utils.read('E:/Dataset/Neurofibromatosis/t12stir_train.yaml')
    train_generator = DataGeneratorGan2d(False, dataset_dict['dataset_list'], dataset_dict['data_shape'], 1, 1, 1)
    data = train_generator.get_data_generator()
    print(next(data))
