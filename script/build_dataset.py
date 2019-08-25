from os import listdir
from utils import yaml_utils, nii_utils
from sklearn.model_selection import train_test_split

a = 'T1'
b = 'STIR'

source = 'E:/Datasets/Neurofibromatosis/source'  #  /home/yf/datas/NF/
output = 'E:/Datasets/Neurofibromatosis'  # E:/Datasets/Neurofibromatosis /home/yf/datas/NF/
o_A = listdir(source + '/' + a)
o_B = listdir(source + '/' + b)

A = list()
B = list()

for i in range(len(o_A)):
    image_A = nii_utils.nii_reader(source + '/' + a + '/' + o_A[i])
    image_B = nii_utils.nii_reader(source + '/' + b + '/' + o_B[i])
    if image_A.shape[2] == image_B.shape[2]:
        A.append(source + '/' + a + '/' + o_A[i])
        B.append(source + '/' + b + '/' + o_B[i])

train_A, test_A, train_B, test_B = train_test_split(A, B, test_size=0.2, random_state=10)

fold = dict()
fold['A'] = train_A
fold['B'] = train_B
yaml_utils.write(output + '/' + a + '2' + b + '_train.yaml', fold)
fold['A'] = test_A
fold['B'] = test_B
yaml_utils.write(output + '/' + a + '2' + b + '_test.yaml', fold)
