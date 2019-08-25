import numpy as np
from os import listdir
from pathlib import Path
from utils import yaml_utils, nii_utils

a = 'T1'
b = 'STIR'

source = '/home/yf/datas/NF/'  # E:/Datasets/Neurofibromatosis/source
output = '/home/yf/datas/NF/'  # E:/Datasets/Neurofibromatosis
oa = listdir(source + '/' + a)
ob = listdir(source + '/' + b)

datasets = list()
error = list()
for i in range(len(oa)):
    image_a = nii_utils.nii_reader(source + '/' + a + '/' + oa[i])
    image_b = nii_utils.nii_reader(source + '/' + b + '/' + ob[i])
    print('\r>>dataset {} name: {}'.format(i, oa[i]), end='')
    if image_a.shape[2] == image_b.shape[2] and oa[i] == ob[i]:
        path = output + 'dataset/' + oa[i] + '.npz'
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, a=image_a, b=image_b, a_size=[400], b_size=[1500])
        datasets.append(path)
    else:
        error.append(
            'dataset {:3d} error, {}:{} {}, {}:{} {}'.format(i, a, oa[i], image_a.shape, b, ob[i], image_b.shape))
        print('\r dataset {:3d} error, {}:{} {}, {}:{} {}'.format(i, a, oa[i], image_a.shape, b, ob[i], image_b.shape))
np.random.shuffle(datasets)

train_datasets = datasets[:len(datasets) * 8 // 10]
print('\r! train dataset size: {}'.format(len(train_datasets)))
yaml_utils.write(output + '/' + a + '2' + b + '_train.yaml', train_datasets)

eval_datasets = datasets[len(train_datasets): len(datasets) * 9 // 10]
print('\r! eval dataset size: {}'.format(len(eval_datasets)))
yaml_utils.write(output + '/' + a + '2' + b + '_eval.yaml', eval_datasets)

test_datasets = datasets[len(train_datasets) + len(eval_datasets):]
print('\r! test dataset size: {}'.format(len(test_datasets)))
yaml_utils.write(output + '/' + a + '2' + b + '_test.yaml', test_datasets)

yaml_utils.write(output + '/error.yaml', error)
