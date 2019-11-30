import numpy as np
from scipy import ndimage

from data_loader import preprocess
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils
from utils.nii_utils import nii_header_reader, nii_writer


class DataGeneratorGan3d(BaseDataGenerator):
    def __init__(self, is_augmented, dataset_list, data_shape, batch_size, in_channels, out_channels,
                 base_patch=(32, 32, 32), **kwargs):
        BaseDataGenerator.__init__(self, is_augmented, dataset_list, data_shape, batch_size, in_channels, out_channels)
        self.base_patch = base_patch

    def get_size(self):
        return len(self.dataset_list)

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

                # d_s, d_e, h_s, h_e, w_s, w_e = self.get_cube_point(nii_a.shape)
                # # d=depth, h=height, w=width,,s=start e=end
                # patch_a = nii_a[d_s:d_e, h_s:h_e, w_s:w_e]
                # patch_b = nii_b[d_s:d_e, h_s:h_e, w_s:w_e]

                patch_a, patch_b = self.get_multi_channel_image(nii_a, self.in_channels, nii_b, self.out_channels)
                batch_a.append(patch_a)
                batch_b.append(patch_b)
                if len(batch_a) == self.batch_size:
                    yield path_a, source_path_a, np.array(batch_a), path_b, source_path_b, np.array(batch_b)
                    batch_a = list()
                    batch_b = list()
                if len(batch_a) > 0:
                    for _ in range(self.batch_size - len(batch_a)):  # one batch(all patch in the same images)
                        batch_a.append(
                            np.zeros((self.base_patch[0], self.base_patch[1], self.base_patch[2], self.in_channels),
                                     dtype=float))
                        batch_b.append(
                            np.zeros((self.base_patch[0], self.base_patch[1], self.base_patch[2], self.out_channels),
                                     dtype=float))
                    yield path_a, source_path_a, np.array(batch_a), path_b, source_path_b, np.array(batch_b)
                    batch_a = list()
                    batch_b = list()

    def get_cube_point(self, nii_shape):
        d_center = np.random.randint(self.base_patch[0] // 2, nii_shape[0] - self.base_patch[0] // 2)
        h_center = np.random.randint(self.base_patch[1] // 2, nii_shape[1] - self.base_patch[1] // 2)
        w_center = np.random.randint(self.base_patch[2] // 2, nii_shape[2] - self.base_patch[2] // 2)
        return d_center - self.base_patch[0] // 2, d_center + self.base_patch[0] - self.base_patch[0] // 2, \
               h_center - self.base_patch[1] // 2, h_center + self.base_patch[1] - self.base_patch[1] // 2, \
               w_center - self.base_patch[2] // 2, w_center + self.base_patch[2] - self.base_patch[2] // 2

    def get_multi_channel_image(self, nii_a, channels_a, nii_b, channels_b):
        channels_images_a = list()
        channels_images_b = list()
        flip = False
        offset_x, offset_y, offset_z = (0, 0, 0)
        d_scale, h_scale, w_scale = (1.0, 1.0, 1.0)
        if self.is_augmented:
            flip = True if np.random.rand() > 0.5 else False
            scale_size = (286, 286, 18)
            offset_z = np.random.randint(0, scale_size[0] - self.data_shape[0])
            offset_y = np.random.randint(0, scale_size[1] - self.data_shape[1])
            offset_x = np.random.randint(0, scale_size[2] - self.data_shape[2])
            d_scale = scale_size[0] * 1.0 / self.data_shape[0]
            h_scale = scale_size[1] * 1.0 / self.data_shape[1]
            w_scale = scale_size[2] * 1.0 / self.data_shape[2]
        if self.is_augmented:
            nii_a = ndimage.interpolation.zoom(nii_a, (d_scale, h_scale, w_scale), order=0)
            nii_a = nii_a[offset_z:offset_z + self.data_shape[0], offset_y:offset_y + self.data_shape[1],
                    offset_x:offset_x + self.data_shape[2]]
            if flip:
                nii_a = np.fliplr(nii_a)
        channels_images_a.append(nii_a)
        if self.is_augmented:
            nii_b = ndimage.interpolation.zoom(nii_b, (d_scale, h_scale, w_scale), order=0)
            nii_b = nii_b[offset_z:offset_z + self.data_shape[0], offset_y:offset_y + self.data_shape[1],
                    offset_x:offset_x + self.data_shape[2]]
            if flip:
                nii_b = np.fliplr(nii_b)
        channels_images_b.append(nii_b)

        channels_images_a = np.array(channels_images_a).transpose((1, 2, 3, 0))
        channels_images_b = np.array(channels_images_b).transpose((1, 2, 3, 0))
        return channels_images_a, channels_images_b


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Dataset/Neurofibromatosis/T12STIR_train.yaml')
    data_loader = DataGeneratorGan3d(dataset, 1, 1, True)
    generator = data_loader.get_data_generator()
    path_a, batch_a, path_b, batch_b = next(generator)
    nii_b_head = nii_header_reader(path_b)
    nii_writer('./fake.nii', nii_b_head, np.squeeze(batch_b))
