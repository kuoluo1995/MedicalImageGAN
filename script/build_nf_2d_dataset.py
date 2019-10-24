import numpy as np
from os import listdir
from pathlib import Path
from scipy import ndimage
from utils import yaml_utils, nii_utils

A = 'T1'
B = 'STIR'

data_shape = [1152, 384]  # [1088,320]

source = '/home/yf/datas/NF'  # /home/yf/datas/NF E:/Dataset/Neurofibromatosis/source
output = '/home/yf/datas/NF'  # /home/yf/datas/NF E:/Dataset/Neurofibromatosis


def drop_invalid_range(data_):
    no_zero_idxs = np.where(data_ > 0)
    [max_d, max_h, max_w] = np.max(np.array(no_zero_idxs), axis=1)
    [min_d, min_h, min_w] = np.min(np.array(no_zero_idxs), axis=1)
    return [max_d, max_h, max_w], [min_d, min_h, min_w]


def crop_data(data_, min_shape, max_shape):
    [min_d, min_h, min_w] = min_shape
    [max_d, max_h, max_w] = max_shape
    return data_[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1]


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
    data_ = np.pad(data_, ((0, d_pad), (0, h_pad), (0, 0)), 'constant')
    data_ = ndimage.interpolation.zoom(data_, (d_scale, h_scale, 1.0), order=0)
    return data_


def normalize_itensity(data_):
    # normalize each slice
    for s_id in range(data_.shape[2]):
        pixels = data_[:, :, s_id]
        pixels = pixels[pixels > 5]
        mean = pixels.mean()
        std = pixels.std()
        data_[:, :, s_id] = (data_[:, :, s_id] - mean) / std
    # normalize to [0,1]
    max_value = np.max(data_)
    min_value = np.min(data_)
    data_ = (data_ - min_value) / (max_value - min_value)
    return data_


def processed_data():
    source_dirs_a = listdir(source + '/' + A)
    source_dirs_b = listdir(source + '/' + B)
    dataset = list()
    dataset_info = list()
    error_data_info = list()
    for i in range(len(source_dirs_a)):
        nii_a = nii_utils.nii_reader(source + '/' + A + '/' + source_dirs_a[i])
        nii_b = nii_utils.nii_reader(source + '/' + B + '/' + source_dirs_b[i])

        nii_a = np.transpose(nii_a, (1, 0, 2))
        nii_b = np.transpose(nii_b, (1, 0, 2))
        if nii_a.shape[2] == nii_b.shape[2] and source_dirs_a[i] == source_dirs_b[i]:
            # drop out the invalid range
            max_shape_a, min_shape_a = drop_invalid_range(nii_a)
            max_shape_b, min_shape_b = drop_invalid_range(nii_b)
            min_shape = np.max([min_shape_a, min_shape_b], axis=0)
            max_shape = np.min([max_shape_a, max_shape_b], axis=0)

            # crop data
            nii_a = crop_data(nii_a, min_shape, max_shape)
            nii_b = crop_data(nii_b, min_shape, max_shape)

            # resize data
            nii_a = resize_data(nii_a)
            nii_b = resize_data(nii_b)

            # normalization data
            nii_a = normalize_itensity(nii_a)
            nii_b = normalize_itensity(nii_b)

            print('\r>>dataset {}/{} name: {}'.format(i + 1, len(source_dirs_a), source_dirs_a[i]), end='')
            path_a = output + '/2d/' + Path(source_dirs_a[i]).stem + '/' + A + '.nii'
            path_b = output + '/2d/' + Path(source_dirs_b[i]).stem + '/' + B + '.nii'
            Path(path_a).parent.mkdir(parents=True, exist_ok=True)
            Path(path_b).parent.mkdir(parents=True, exist_ok=True)
            header_a = nii_utils.nii_header_reader(source + '/' + A + '/' + source_dirs_a[i])
            header_b = nii_utils.nii_header_reader(source + '/' + B + '/' + source_dirs_b[i])
            nii_utils.nii_writer(path_a, header_a, nii_a)
            nii_utils.nii_writer(path_b, header_b, nii_b)
            dataset_path = output + '/2d/' + Path(source_dirs_a[i]).stem + '.npz'
            Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(dataset_path, A=nii_a, B=nii_b, path_a=path_a, path_b=path_b,
                     source_path_a=source + '/' + A + '/' + source_dirs_a[i],
                     source_path_b=source + '/' + B + '/' + source_dirs_b[i])
            dataset.append(dataset_path)
            dataset_info.append(
                '{} {}:{},{}:{}'.format(str(source_dirs_a[i]), A, tuple(nii_a.shape), B, tuple(nii_b.shape)))
        else:
            error_data_info.append('{}:{} {},{}:{} {}'.format(A, source_dirs_a[i], nii_a.shape, B, source_dirs_b[i],
                                                              nii_b.shape))
            print('\r dataset error, {}:{} {},{}:{} {}'.format(A, source_dirs_a[i], nii_a.shape, B, source_dirs_b[i],
                                                               nii_b.shape))
    np.random.shuffle(dataset)
    train_dataset = dataset[:len(dataset) * 8 // 10]
    print('\r! train dataset size: {}'.format(len(train_dataset)))
    train_dict = {'data_shape': data_shape, 'dataset_list': train_dataset}
    yaml_utils.write(output + '/train_2d_' + A + '2' + B + '.yaml', train_dict)

    eval_dataset = dataset[len(train_dataset): len(dataset) * 9 // 10]
    print('\r! eval dataset size: {}'.format(len(eval_dataset)))
    eval_dict = {'data_shape': data_shape, 'dataset_list': eval_dataset}
    yaml_utils.write(output + '/eval_2d_' + A + '2' + B + '.yaml', eval_dict)

    test_dataset = dataset[len(dataset) * 9 // 10:]
    print('\r! test dataset size: {}'.format(len(test_dataset)))
    test_dict = {'data_shape': data_shape, 'dataset_list': test_dataset}
    yaml_utils.write(output + '/test_2d_' + A + '2' + B + '.yaml', test_dict)

    yaml_utils.write(output + '/error_2d.yaml', error_data_info)
    yaml_utils.write(output + '/dataset_info_2d.yaml', dataset_info)


if __name__ == '__main__':
    processed_data()
