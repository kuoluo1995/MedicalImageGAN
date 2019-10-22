# coding=utf-8
import cv2
import numpy as np
from utils import nii_utils

nii_array = nii_utils.nii_reader('E:/Dataset/Neurofibromatosis/processed_dataset/2d/2645145/STIR.nii')
# 统计每个灰度下的像素个数
hist = dict()
for i in range(nii_array.shape[0]):
    for j in range(nii_array.shape[1]):
        for k in range(nii_array.shape[2]):
            value = nii_array[i, j, k]
            if value in hist:
                hist[value] += 1
            else:
                hist[value] = 0
# 统计灰度频率
hist_list = list()
for k, v in hist.items():
    hist_list.append((k, v / (nii_array.shape[0] * nii_array.shape[1] * nii_array.shape[2])))


def sort_key(item):
    return item[0]


hist_list.sort(key=sort_key)

# 计算累积密度
gray_prob = list()
pre_val = 0
for i, v in hist_list:
    gray_prob.append((v[0], v[1] + pre_val))
    pre_val = v[1] + pre_val
# 重新计算均衡后的灰度
for 

image = cv2.imread("E:/timg.jpg", 0)

lut = np.zeros(256, dtype=image.dtype)  # 创建空的查找表

hist, bins = np.histogram(image.flatten(), 256, [0, 256])
cdf = hist.cumsum()  # 计算累积直方图
cdf_m = np.ma.masked_equal(cdf, 0)  # 除去直方图中的0值
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # 等同于前面介绍的lut[i] = int(255.0 *p[i])公式
cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # 将掩模处理掉的元素补为0

# 计算
result2 = cdf[image]
result = cv2.LUT(image, cdf)

cv2.imshow("OpenCVLUT", result)
cv2.imshow("NumPyLUT", result2)
cv2.waitKey(0)
cv2.destroyAllWindows()
