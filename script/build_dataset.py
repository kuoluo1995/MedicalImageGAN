import numpy as np
from os import listdir
from pathlib import Path
from utils import yaml_utils, nii_utils

A = 'T1'
B = 'STIR'

source = 'E:/Datasets/Neurofibromatosis/source'  # /home/yf/datas/NF
output = 'E:/Datasets/Neurofibromatosis'  # /home/yf/datas/NF

sourceA_dirs = listdir(source + '/' + A)
sourceB_dirs = listdir(source + '/' + B)

dataset = list()
error_data_info = list()
for i in range(len(sourceA_dirs)):
    a_nii = nii_utils.nii_reader(source + '/' + A + '/' + sourceA_dirs[i])
    b_nii = nii_utils.nii_reader(source + '/' + B + '/' + sourceB_dirs[i])
    print('\r>>dataset {}/{} name: {}'.format(i, len(sourceA_dirs), sourceA_dirs[i]), end='')
    if a_nii.shape[2] == b_nii.shape[2] and sourceA_dirs[i] == sourceB_dirs[i]:
        path = output + '/dataset/' + sourceA_dirs[i] + '.npz'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, A=a_nii, max_valueA=[1500], B=b_nii, max_valueB=[400])
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
