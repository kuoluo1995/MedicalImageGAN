import math
import numpy as np
from utils import nii_utils


def distance(i, j, k, x, y, z):
    return np.sqrt((x - i) ** 2 + (y - j) ** 2 + (z - k) ** 2)


def gaussian(x, sigma):
    return (1.0 / 2 * math.pi * (sigma ** 2)) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def _bilateral_filter(image, x, y, z, filter_size, sigma_i, sigma_s):
    hl = filter_size // 2
    values = 0
    weights = 0
    for k in range(filter_size):
        for j in range(filter_size):
            for i in range(filter_size):
                neighbour_x = x - (hl - i)
                neighbour_y = y - (hl - j)
                neighbour_z = z - (hl - k)
                wd = gaussian(distance(neighbour_x, neighbour_y, neighbour_z, x, y, z), sigma_s)  # 空间域
                if neighbour_z >= image.shape[0] or neighbour_z < 0 or neighbour_y >= image.shape[
                    1] or neighbour_y < 0 or neighbour_x >= image.shape[2] or neighbour_x < 0:
                    weight = 0
                    values += 0
                else:
                    wr = gaussian(image[neighbour_z, neighbour_y, neighbour_x] - image[z, y, x], sigma_i)  # 值域
                    weight = wr * wd
                    values += image[neighbour_z, neighbour_y, neighbour_x] * weight
                weights += weight
    values = values / weights
    return values


def bilateral_filter(image, filter_size, sigma_i, sigma_s):
    filtered_image = np.zeros(image.shape)
    for z in range(image.shape[0]):
        for y in range(image.shape[1]):
            for x in range(image.shape[2]):
                filtered_image[z, y, x] = _bilateral_filter(image, x, y, z, filter_size, sigma_i, sigma_s)
                print('z:{}/{} h:{}/{} w:{}/{}'.format(z + 1, image.shape[0], y + 1, image.shape[1], x + 1,
                                                       image.shape[2]))
    return filtered_image


nii_array = nii_utils.nii_reader('E:/Dataset/Neurofibromatosis/processed_dataset/2d/2645145/STIR.nii')
result = bilateral_filter(nii_array, 5, 12.0, 16.0)
head = nii_utils.nii_header_reader('E:/Dataset/Neurofibromatosis/processed_dataset/2d/2645145/STIR.nii')
nii_utils.nii_writer('E:/Dataset/Neurofibromatosis/processed_dataset/2d/2645145/STIR_f.nii', head, result)
