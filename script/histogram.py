# coding=utf-8
from utils import nii_utils

nii_array = nii_utils.nii_reader('E:/Dataset/Neurofibromatosis/processed_dataset/2d/2645145/STIR_f.nii')
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
pre_val = 0
for k, v in hist_list:
    pre_val = v + pre_val
    hist[k] = pre_val
# 重新计算均衡后的灰度
for i in range(nii_array.shape[0]):
    for j in range(nii_array.shape[1]):
        for k in range(nii_array.shape[2]):
            nii_array[i, j, k] = hist[nii_array[i, j, k]]
head = nii_utils.nii_header_reader('E:/Dataset/Neurofibromatosis/processed_dataset/2d/2645145/STIR_f.nii')
nii_utils.nii_writer('E:/Dataset/Neurofibromatosis/processed_dataset/2d/2645145/STIR_fh.nii', head, nii_array)
