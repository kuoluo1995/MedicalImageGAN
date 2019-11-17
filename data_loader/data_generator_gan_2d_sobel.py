import cv2
import math
import numpy as np

from data_loader import preprocess
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils


class DataGeneratorGan2dSobel(BaseDataGenerator):
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

        if self.is_augmented:
            flip = True if np.random.rand() > 0.5 else False
            scale_size = (286, 286)
            slice_a = cv2.resize(nii_a[:, :, s_id], scale_size, interpolation=cv2.INTER_AREA)
            slice_b = cv2.resize(nii_b[:, :, s_id], scale_size, interpolation=cv2.INTER_AREA)
            offset_y = np.random.randint(0, scale_size[0] - self.data_shape[0])
            offset_x = np.random.randint(0, scale_size[1] - self.data_shape[1])
            slice_a = slice_a[offset_y:offset_y + self.data_shape[0], offset_x:offset_x + self.data_shape[1]]
            slice_b = slice_b[offset_y:offset_y + self.data_shape[0], offset_x:offset_x + self.data_shape[1]]
            if flip:
                slice_a = np.fliplr(slice_a)
                slice_b = np.fliplr(slice_b)
        else:
            slice_a = nii_a[:, :, s_id]
            slice_b = nii_b[:, :, s_id]

        channels_images_a.append(slice_a)
        sobel_x = cv2.Sobel(slice_a, cv2.CV_64F, 1, 0, ksize=9)
        channels_images_a.append(sobel_x)
        sobel_y = cv2.Sobel(slice_a, cv2.CV_64F, 0, 1, ksize=9)
        channels_images_a.append(sobel_y)
        sobel_xy = cv2.Sobel(slice_a, cv2.CV_64F, 1, 1, ksize=9)
        channels_images_a.append(sobel_xy)

        channels_images_b.append(slice_b)

        channels_images_a = np.array(channels_images_a).transpose((1, 2, 0))
        channels_images_b = np.array(channels_images_b).transpose((1, 2, 0))
        return channels_images_a, channels_images_b


if __name__ == '__main__':
    dataset_dict = yaml_utils.read('E:/Dataset/Neurofibromatosis/t12stir_train.yaml')
    train_generator = DataGeneratorGan2d(False, dataset_dict['dataset_list'], dataset_dict['data_shape'], 1, 1, 1)
    data = train_generator.get_data_generator()
    print(next(data))
