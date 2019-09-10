import numpy as np
from os import listdir
from pathlib import Path
from utils import yaml_utils, nii_utils

A = 'T1'
B = 'STIR'

source = '/home/yf/datas/NF'  # /home/yf/datas/NF E:/Datasets/Neurofibromatosis/source
output = '/home/yf/datas/NF'  # E:/Datasets/Neurofibromatosis

sourceA_dirs = listdir(source + '/' + A)
sourceB_dirs = listdir(source + '/' + B)

dataset = list()
error_data_info = list()
for i in range(len(sourceA_dirs)):
    a_nii = nii_utils.nii_reader(source + '/' + A + '/' + sourceA_dirs[i])
    a_nii = np.transpose(a_nii, (2, 1, 0))
    a_nii = np.clip(a_nii, 0, 1500) / 1500
    b_nii = nii_utils.nii_reader(source + '/' + B + '/' + sourceB_dirs[i])
    b_nii = np.transpose(b_nii, (2, 1, 0))
    b_nii = np.clip(b_nii, 0, 400) / 400
    print('\r>>dataset {}/{} name: {}'.format(i, len(sourceA_dirs), sourceA_dirs[i]), end='')
    if a_nii.shape[0] == b_nii.shape[0] and sourceA_dirs[i] == sourceB_dirs[i]:
        path = output + '/dataset/' + sourceA_dirs[i] + '.npz'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, A=a_nii, B=b_nii, A_path=source + '/' + A + '/' + sourceA_dirs[i],
                 B_path=source + '/' + B + '/' + sourceB_dirs[i])
        dataset.append(path)
    else:
        error_data_info.append(
            'dataset error, {}-name:{} shape:{}, {}-name:{} shape:{}'.format(A, sourceA_dirs[i], a_nii.shape, B,
                                                                             sourceB_dirs[i], b_nii.shape))
        print('\r dataset error, {}-name:{} shape:{}, {}-name:{} shape:{}'.format(A, sourceA_dirs[i], a_nii.shape, B,
                                                                                  sourceB_dirs[i], b_nii.shape))
np.random.shuffle(dataset)
train_dataset = dataset[:len(dataset) * 8 // 10]
print('\r! train dataset size: {}'.format(len(train_dataset)))
yaml_utils.write(output + '/' + A + '2' + B + '_train.yaml', train_dataset)

eval_dataset = dataset[len(train_dataset): len(dataset) * 9 // 10]
print('\r! eval dataset size: {}'.format(len(eval_dataset)))
yaml_utils.write(output + '/' + A + '2' + B + '_eval.yaml', eval_dataset)

test_dataset = dataset[len(dataset) * 9 // 10:]
print('\r! test dataset size: {}'.format(len(test_dataset)))
yaml_utils.write(output + '/' + A + '2' + B + '_test.yaml', test_dataset)

yaml_utils.write(output + '/error.yaml', error_data_info)
