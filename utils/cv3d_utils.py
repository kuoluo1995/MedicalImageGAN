import cv2
import math
import numpy as np
import SimpleITK as sitk
from utils import nii_utils
# from PIL import Image, ImageFilter


# 直方图均衡化
def histogram_equalization_3d(nii_array):
    # 统计每个灰度下的像素个数
    histogram = dict()
    for i in range(nii_array.shape[0]):
        for j in range(nii_array.shape[1]):
            for k in range(nii_array.shape[2]):
                value = nii_array[i, j, k]
                if value in histogram:
                    histogram[value] += 1
                else:
                    histogram[value] = 0
    histogram_list = list()
    for k, v in histogram.items():
        histogram_list.append((k, v / (nii_array.shape[0] * nii_array.shape[1] * nii_array.shape[2])))

    def sort_key(item):
        return item[0]

    histogram_list.sort(key=sort_key)
    # 统计灰度频率 计算累积密度
    accumulation = 0
    for k, v in histogram_list:
        accumulation = v + accumulation
        histogram[k] = accumulation
    # 重新计算均衡后的灰度
    for i in range(nii_array.shape[0]):
        for j in range(nii_array.shape[1]):
            for k in range(nii_array.shape[2]):
                nii_array[i, j, k] = histogram[nii_array[i, j, k]]
    return nii_array


def bilateral_filter_3d_itk(nii_array, filter_size, domain_sigma, range_sigma):
    sitk_image = sitk.GetImageFromArray(nii_array)
    _bilateral_filter = sitk.BilateralImageFilter()
    sitk_image = _bilateral_filter.Execute(sitk_image, domain_sigma, range_sigma, filter_size)
    nii_array = sitk.GetArrayFromImage(sitk_image)
    return nii_array


# 双边滤波算法
def bilateral_filter_3d(nii_array, filter_size, sigma_value, sigma_space):
    def _distance(z_a, y_a, x_a, z_b, y_b, x_b):
        return np.sqrt((x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2)

    def _gaussian(x, sigma):
        return (1.0 / 2 * math.pi * (sigma ** 2)) * math.exp(- (x ** 2) / (2 * sigma ** 2))

    def _bilateral_filter(center_z, center_y, center_x):
        value = 0
        weight = 0
        radius = filter_size // 2
        for k in range(filter_size):
            for j in range(filter_size):
                for i in range(filter_size):
                    neighbour_x = center_x - (radius - i)
                    neighbour_y = center_y - (radius - j)
                    neighbour_z = center_z - (radius - k)
                    # 计算空间域权重
                    weight_distance = _gaussian(
                        _distance(neighbour_z, neighbour_y, neighbour_x, center_z, center_y, center_x), sigma_space)
                    if 0 <= neighbour_z < nii_array.shape[0] and 0 <= neighbour_y < nii_array.shape[
                        1] and 0 <= neighbour_x < nii_array.shape[2]:
                        # 计算值域权重
                        weight_value = _gaussian(
                            nii_array[neighbour_z, neighbour_y, neighbour_x] - nii_array[center_z, center_y, center_x],
                            sigma_value)
                        w_k = weight_value * weight_distance
                        value += nii_array[neighbour_z, neighbour_y, neighbour_x] * w_k
                        weight += w_k
        value = value / weight
        return value

    filtered_nii = np.zeros(nii_array.shape)
    for z in range(nii_array.shape[0]):
        for y in range(nii_array.shape[1]):
            for x in range(nii_array.shape[2]):
                filtered_nii[z, y, x] = _bilateral_filter(z, y, x)
    return filtered_nii


# 伽玛变换
def gamma_transform(nii_array, percent):
    nii_array = pow(nii_array, percent)
    return nii_array


# 二维的双边滤波
def bilateral_filter_2d(nii_array, filter_size=3, domain_sigma=100, range_sigma=15):
    sitk_image = sitk.GetImageFromArray(nii_array)
    _bilateral_filter = sitk.BilateralImageFilter()
    for i in range(10):
        sitk_image = _bilateral_filter.Execute(sitk_image, domain_sigma, range_sigma, filter_size)
    nii_array = sitk.GetArrayFromImage(sitk_image)
    return nii_array

if __name__ == '__main__':
    # 直方图均衡化
    nii_array = nii_utils.nii_reader('E:/Dataset/Neurofibromatosis/2d_patch/WBMRI_009/STIR.nii')
    # nii_array = histogram_equalization_3d(nii_array)
    # nii_array2 = bilateral_filter_3d_itk(nii_array, 5, 12.0, 16.0)
    # nii_array = cv2.bilateralFilter(nii_array, 0, 10, 10)
    # nii_array = cv2.GaussianBlur(nii_array, (5, 5), 10)
    nii_array = cv2.Sobel(nii_array, cv2.CV_64F, 2, 2, ksize=5)
    head = nii_utils.nii_header_reader('E:/Dataset/Neurofibromatosis/2d_patch/WBMRI_009/STIR.nii')
    nii_utils.nii_writer('E:/Dataset/Neurofibromatosis/2d_patch/WBMRI_009/STIR_canny.nii', head, nii_array)
    # 双边滤波算法
