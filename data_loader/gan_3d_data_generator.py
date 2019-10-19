import numpy as np
from data_loader.base_data_generator import BaseDataGenerator
from utils import yaml_utils
from utils.nii_utils import nii_header_reader, nii_writer


class Gan3dDataGenerator(BaseDataGenerator):
    def __init__(self, dataset_list, batch_size, image_size, in_channels, out_channels, is_training,
                 base_patch=(32, 32, 32), **kwargs):
        BaseDataGenerator.__init__(self, dataset_list, batch_size, image_size, in_channels, out_channels, is_training,
                                   base_patch)

    def get_size(self):
        return len(self.dataset_list)

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
                d_s, d_d, h_s, h_d, w_s, w_d = self.get_cube_point(a_nii.shape)
                # d=depth, h=height, w=width,,s=start d=end
                a_patch = a_nii[d_s:d_d + 1, h_s:h_d + 1, w_s:w_d + 1]
                b_patch = b_nii[d_s:d_d + 1, h_s:h_d + 1, w_s:w_d + 1]
                a_patch, b_patch = self.get_multi_channel_image(a_patch, self.in_channels, b_patch, self.out_channels)
                batchA.append(a_patch)
                batchB.append(b_patch)
                if len(batchA) == self.batch_size:
                    yield a_path, np.array(batchA), b_path, np.array(batchB)
                    batchA = list()
                    batchB = list()
                if len(batchA) > 0:
                    for _ in range(self.batch_size - len(batchA)):  # one batch(all patch in the same images)
                        batchA.append(
                            np.zeros((self.base_patch[0], self.base_patch[1], self.base_patch[2], self.in_channels),
                                     dtype=float))
                        batchB.append(
                            np.zeros((self.base_patch[0], self.base_patch[1], self.base_patch[2], self.out_channels),
                                     dtype=float))
                    yield a_path, np.array(batchA), b_path, np.array(batchB)
                    batchA = list()
                    batchB = list()

    def get_cube_point(self, nii_shape):
        d_center = np.random.randint(self.base_patch[0] // 2, nii_shape[0] - self.base_patch[0] // 2)
        h_center = np.random.randint(self.base_patch[1] // 2, nii_shape[1] - self.base_patch[1] // 2)
        w_center = np.random.randint(self.base_patch[2] // 2, nii_shape[2] - self.base_patch[2] // 2)
        return d_center - self.base_patch[0] // 2, d_center + self.base_patch[0] - self.base_patch[0] // 2, h_center - \
               self.base_patch[1] // 2, h_center + self.base_patch[1] - self.base_patch[1] // 2, w_center - \
               self.base_patch[2] // 2, w_center + self.base_patch[2] - self.base_patch[2] // 2

    def get_multi_channel_image(self, a_data, a_channels, b_data, b_channels):
        itensity = 0
        if self.is_training:  # todo 数据增广
            itensity = np.random.rand() * 0.1
        channels_images_a = list()
        channels_images_b = list()
        channels_images_a.append(a_data + itensity)
        channels_images_b.append(b_data + itensity)
        channels_images_a = np.array(channels_images_a).transpose((1, 2, 3, 0))
        channels_images_b = np.array(channels_images_b).transpose((1, 2, 3, 0))
        return channels_images_a, channels_images_b


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Dataset/Neurofibromatosis/T12STIR_train.yaml')
    data_loader = Gan3dDataGenerator(dataset, 1, 1, True)
    generator = data_loader.get_data_generator()
    a_path, batchA, b_path, batchB = next(generator)
    b_nii_head = nii_header_reader(b_path)
    nii_writer('./fake.nii', b_nii_head, np.squeeze(batchB))
