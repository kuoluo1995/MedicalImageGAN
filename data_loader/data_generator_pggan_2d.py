import numpy as np
from scipy import ndimage

from data_loader import preprocess
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils


class DataGeneratorPGGan2d(BaseDataGenerator):
    def __init__(self, is_augmented, dataset_list, data_shape, batch_size, in_channels, out_channels, **kwargs):
        BaseDataGenerator.__init__(self, is_augmented, dataset_list, data_shape, batch_size, in_channels, out_channels)
        self.scale_size = (286, 286)
        self.size = 0
        for item in self.dataset_list:
            npz = np.load(item)
            self.size += npz['A'].shape[2]

    def get_size(self):
        return self.size // self.batch_size

    def get_batch_size(self):
        return self.batch_size

    def get_in_channels(self):
        return self.in_channels

    def get_out_channels(self):
        return self.out_channels

    def resize_data(self, nii, h_size, w_size):
        h_scale = h_size * 1.0 / self.data_shape[0]
        w_scale = w_size * 1.0 / self.data_shape[1]
        _nii = ndimage.interpolation.zoom(nii, (h_scale, w_scale, 1.0), order=0)
        return _nii

    def get_data_generator(self, height=256, width=256):
        while True:
            batch_a = list()
            batch_b = list()
            if self.is_augmented:
                np.random.shuffle(self.dataset_list)
            for item in self.dataset_list:
                npz = np.load(item)
                nii_a = preprocess(npz['A'])  # [0, 1] => [-1, 1]
                nii_b = preprocess(npz['B'])  # [0, 1] => [-1, 1]
                # path_a is preprocessed dataset path, source_path is pre-processed dataset path
                path_a, source_path_a = str(npz['path_a']), str(npz['source_path_a'])
                path_b, source_path_b = str(npz['path_b']), str(npz['source_path_b'])
                for s_id in range(nii_a.shape[2]):
                    slice_a, slice_b = self.get_multi_channel_image(s_id, nii_a, self.in_channels, nii_b,
                                                                    self.out_channels, height, width)
                    batch_a.append(slice_a)
                    batch_b.append(slice_b)
                    if len(batch_a) == self.batch_size:
                        yield path_a, source_path_a, np.array(batch_a), path_b, source_path_b, np.array(batch_b)
                        batch_a = list()
                        batch_b = list()

    def get_multi_channel_image(self, s_id, nii_a, channels_a, nii_b, channels_b, height, width):
        channels_images_a = []
        channels_images_b = []
        for _ in range(s_id, channels_a // 2):
            channels_images_a.append(np.zeros(self.data_shape, dtype=float))
        for _ in range(s_id, channels_b // 2):
            channels_images_b.append(np.zeros(self.data_shape, dtype=float))
        padding_a = len(channels_images_a)
        padding_b = len(channels_images_b)

        flip = False
        offset_x, offset_y = (0, 0)
        if self.is_augmented:  # Unified Augmentation
            flip = True if np.random.rand() > 0.5 else False
            offset_y = np.random.randint(0, self.scale_size[0] - self.data_shape[0])
            offset_x = np.random.randint(0, self.scale_size[1] - self.data_shape[1])
        for _id in range(s_id - channels_a // 2 + padding_a, min(s_id + channels_a - channels_a // 2, nii_a.shape[2])):
            slice_a = self.data_augment(nii_a[:, :, _id], offset_y, offset_x, height, width, flip)
            channels_images_a.append(slice_a)
        for _id in range(s_id - channels_b // 2 + padding_b, min(s_id + channels_b - channels_b // 2, nii_b.shape[2])):
            slice_b = self.data_augment(nii_b[:, :, _id], offset_y, offset_x, height, width, flip)
            channels_images_b.append(slice_b)
        padding_a = len(channels_images_a)
        padding_b = len(channels_images_b)

        for _ in range(channels_a - padding_a):
            channels_images_a.append(np.zeros(self.data_shape, dtype=float))
        for _ in range(channels_b - padding_b):
            channels_images_b.append(np.zeros(self.data_shape, dtype=float))

        channels_images_a = np.array(channels_images_a).transpose((1, 2, 0))
        channels_images_b = np.array(channels_images_b).transpose((1, 2, 0))
        return channels_images_a, channels_images_b

    def data_augment(self, _slice, offset_y, offset_x, height, width, flip):
        if self.is_augmented:
            h_scale = self.scale_size[0] * 1.0 / self.data_shape[0]
            w_scale = self.scale_size[1] * 1.0 / self.data_shape[1]
            _slice = ndimage.interpolation.zoom(_slice, (h_scale, w_scale), order=0)
            _slice = _slice[offset_y:offset_y + self.data_shape[0], offset_x:offset_x + self.data_shape[1]]
            h_scale = height * 1.0 / self.data_shape[0]
            w_scale = width * 1.0 / self.data_shape[1]
            _slice = ndimage.interpolation.zoom(_slice, (h_scale, w_scale), order=0)
            if flip:
                _slice = np.fliplr(_slice)
        return _slice


if __name__ == '__main__':
    dataset_dict = yaml_utils.read('E:/Dataset/Neurofibromatosis/train_2d_patch_half_T12STIR.yaml')
    train_generator = DataGeneratorPGGan2d(True, dataset_dict['dataset_list'], dataset_dict['data_shape'], 1, 1, 1)
    data = train_generator.get_data_generator()
    print(next(data))
