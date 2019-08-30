import cv2
import random
import numpy as np

from utils import yaml_utils


def get_data_loader_by_name(name):
    return eval(name)


def gan_data_generator(dataset, batch_size, image_size, channels, is_training=True, **kwargs):
    batchA = list()
    batchB = list()
    while True:
        for item in dataset:
            npz = np.load(item)
            a_nii = npz['A']
            b_nii = npz['B']
            for s_id in range(a_nii.shape[0]):
                a, b = get_multi_channel_image(s_id, a_nii, b_nii, image_size, channels, is_training)
                batchA.append(a)
                batchB.append(b)
                if len(batchA) == batch_size:
                    yield np.array(batchA), np.array(batchB)
                    batchA = list()
                    batchB = list()


def get_epoch_step(dataset):
    count = 0
    for item in dataset:
        npz = np.load(item)
        count += npz['A'].shape[0]
    return count


def get_multi_channel_image(s_id, a_nii, b_nii, image_size, channels, is_training):
    channels_imagesA = []
    channels_imagesB = []
    for _ in range(s_id, channels // 2):
        channels_imagesA.append(np.zeros(image_size, dtype=float))
        channels_imagesB.append(np.zeros(image_size, dtype=float))
    padding = len(channels_imagesA)

    for _id in range(s_id - channels // 2 + padding, min(s_id + channels - channels // 2, a_nii.shape[0])):
        sliceA = cv2.resize(a_nii[_id], (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
        sliceB = cv2.resize(b_nii[_id], (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
        if is_training:
            # todo 数据增广
            pass
        channels_imagesA.append(sliceA)
        channels_imagesB.append(sliceB)
    padding = len(channels_imagesA)

    for _ in range(channels - padding):
        channels_imagesA.append(np.zeros(image_size, dtype=float))
        channels_imagesB.append(np.zeros(image_size, dtype=float))

    channels_imagesA = np.array(channels_imagesA).transpose((1, 2, 0))
    channels_imagesB = np.array(channels_imagesB).transpose((1, 2, 0))
    return channels_imagesA, channels_imagesB


if __name__ == '__main__':
    dataset = yaml_utils.read('E:/Datasets/Neurofibromatosis/t12stir_train.yaml')
    train_generator = gan_data_generator(dataset, 8, (512, 256), 1)
    data = next(train_generator)
