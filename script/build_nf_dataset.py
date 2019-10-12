import numpy as np
from os import listdir
from pathlib import Path
from scipy import ndimage
from utils import yaml_utils, nii_utils

A = 't1'
B = 'stir'

data_shape = [1088, 320, 32]

source = '/home/yf/datas/NF'  # /home/yf/datas/NF E:/Dataset/Neurofibromatosis/source
output = '/home/yf/datas/NF'  # E:/Dataset/Neurofibromatosis /home/yf/datas/NF


def drop_invalid_range(data_):
    zero_value = data_[0, 0, 0]
    no_zero_idxs = np.where(data_ != zero_value)
    [max_d, max_h, max_w] = np.max(np.array(no_zero_idxs), axis=1)
    [min_d, min_h, min_w] = np.min(np.array(no_zero_idxs), axis=1)
    return [max_d, max_h, max_w], [min_d, min_h, min_w]


def crop_data(data_, min_shape, max_shape):
    [min_d, min_h, min_w] = min_shape
    [max_d, max_h, max_w] = max_shape
    return data_[min_d:max_d, min_h:max_h, min_w:max_w]


def resize_data(data_):
    shape = data_.shape
    d_pad = 0
    d_scale = 1.0
    if shape[0] <= data_shape[0]:
        d_pad = data_shape[0] - shape[0]
    else:
        d_scale = data_shape[0] * 1.0 / shape[0]
    h_pad = 0
    h_scale = 1.0
    if shape[1] <= data_shape[1]:
        h_pad = data_shape[1] - shape[1]
    else:
        h_scale = data_shape[1] * 1.0 / shape[1]
    w_pad = 0
    w_scale = 1.0
    if shape[2] <= data_shape[2]:
        w_pad = data_shape[2] - shape[2]
    else:
        w_scale = data_shape[2] * 1.0 / shape[2]
    data_ = np.pad(data_, ((0, d_pad), (0, h_pad), (0, w_pad)), 'constant')
    data_ = ndimage.interpolation.zoom(data_, (d_scale, h_scale, w_scale), order=0)
    return data_


def itensity_normalize_data(data_):
    pixels = data_[data_ > 0]
    mean = pixels.mean()
    std = pixels.std()
    out = (data_ - mean) / std
    return out


def processed_data():
    sourceA_dirs = listdir(source + '/' + A)
    sourceB_dirs = listdir(source + '/' + B)
    dataset = list()
    dataset_info = list()
    error_data_info = list()
    for i in range(len(sourceA_dirs)):
        a_nii = nii_utils.nii_reader(source + '/' + A + '/' + sourceA_dirs[i])
        b_nii = nii_utils.nii_reader(source + '/' + B + '/' + sourceB_dirs[i])

        a_nii = np.transpose(a_nii, (1, 0, 2))
        b_nii = np.transpose(b_nii, (1, 0, 2))

        # drop out the invalid range
        a_max_shape, a_min_shape = drop_invalid_range(a_nii)
        b_max_shape, b_min_shape = drop_invalid_range(b_nii)
        min_shape = np.max([a_min_shape, b_min_shape], axis=0)
        max_shape = np.min([a_max_shape, b_max_shape], axis=0)

        # crop data
        a_nii = crop_data(a_nii, min_shape, max_shape)
        b_nii = crop_data(b_nii, min_shape, max_shape)

        # resize data
        a_nii = resize_data(a_nii)
        b_nii = resize_data(b_nii)

        # normalization data
        a_nii = itensity_normalize_data(a_nii)
        b_nii = itensity_normalize_data(b_nii)

        print('\r>>dataset {}/{} name: {}'.format(i, len(sourceA_dirs), sourceA_dirs[i]), end='')
        if a_nii.shape[2] == b_nii.shape[2] and sourceA_dirs[i] == sourceB_dirs[i]:
            a_path = output + '/processed_dataset/' + Path(sourceA_dirs[i]).stem + '/' + A + '.nii'
            b_path = output + '/processed_dataset/' + Path(sourceB_dirs[i]).stem + '/' + B + '.nii'
            Path(a_path).parent.mkdir(parents=True, exist_ok=True)
            Path(b_path).parent.mkdir(parents=True, exist_ok=True)
            a_header = nii_utils.nii_header_reader(source + '/' + A + '/' + sourceA_dirs[i])
            b_header = nii_utils.nii_header_reader(source + '/' + B + '/' + sourceB_dirs[i])
            nii_utils.nii_writer(a_path, a_header, a_nii)
            nii_utils.nii_writer(b_path, b_header, b_nii)
            dataset_path = output + '/dataset/' + Path(sourceA_dirs[i]).stem + '.npz'
            Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(dataset_path, A=a_nii, B=b_nii, A_path=a_path, B_path=b_path)
            dataset.append(dataset_path)
            dataset_info.append(
                'path:{} {}_shape:{}, {}_shape:{}'.format(str(sourceA_dirs[i]), A, tuple(a_nii.shape), B,
                                                          tuple(b_nii.shape)))
        else:
            error_data_info.append(
                'dataset error, {}-name:{} shape:{}, {}-name:{} shape:{}'.format(A, sourceA_dirs[i], a_nii.shape, B,
                                                                                 sourceB_dirs[i], b_nii.shape))
            print(
                '\r dataset error, {}-name:{} shape:{}, {}-name:{} shape:{}'.format(A, sourceA_dirs[i], a_nii.shape, B,
                                                                                    sourceB_dirs[i], b_nii.shape))
    np.random.shuffle(dataset)
    train_dataset = dataset[:len(dataset) * 8 // 10]
    print('\r! train dataset size: {}'.format(len(train_dataset)))
    train_dict = {'shape': data_shape, 'dataset': train_dataset}
    yaml_utils.write(output + '/' + A + '2' + B + '_train.yaml', train_dict)

    eval_dataset = dataset[len(train_dataset): len(dataset) * 9 // 10]
    print('\r! eval dataset size: {}'.format(len(eval_dataset)))
    eval_dict = {'shape': data_shape, 'dataset': eval_dataset}
    yaml_utils.write(output + '/' + A + '2' + B + '_eval.yaml', eval_dict)

    test_dataset = dataset[len(dataset) * 9 // 10:]
    print('\r! test dataset size: {}'.format(len(test_dataset)))
    test_dict = {'shape': data_shape, 'dataset': test_dataset}
    yaml_utils.write(output + '/' + A + '2' + B + '_test.yaml', test_dict)

    yaml_utils.write(output + '/error.yaml', error_data_info)
    yaml_utils.write(output + '/dataset_info.yaml', dataset_info)


if __name__ == '__main__':
    processed_data()
