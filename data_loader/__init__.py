import numpy as np
from utils import yaml_utils


def get_data_generator_fn_by_name(name):
    return eval(name)


def gan_data_generator(dataset_path, batch_size, image_size, channels, is_train=True, **kwargs):
    dataset = yaml_utils.read(dataset_path)
    batch_data_a = list()
    batch_data_b = list()
    while True:
        for item in dataset:
            data = np.load(item)
            data_a = np.transpose(data['a'], (2, 1, 0))
            data_a = np.clip(data_a, 0, data['a_size'][0]) / data['a_size'][0]
            data_b = np.transpose(data['b'], (2, 1, 0))
            data_b = np.clip(data_b, 0, data['b_size'][0]) / data['b_size'][0]
            for i in range(data_a.shape[0]):
                data_a_channels = list()
                data_b_channels = list()
                for _ in range(i, channels // 2):
                    data_a_channels.append(np.zeros(image_size, dtype=float))
                    data_b_channels.append(np.zeros(image_size, dtype=float))
                padding = len(data_a_channels)
                for j in range(i - channels // 2 + padding, min(i + channels // 2, data_a.shape[0])):
                    a = np.resize(data_a[i], (image_size[0], image_size[1]))
                    data_a_channels.append(a)
                    b = np.resize(data_b[i], (image_size[0], image_size[1]))
                    data_b_channels.append(b)
                padding = len(data_a_channels)
                for _ in range(channels - padding):
                    data_a_channels.append(np.zeros(image_size, dtype=float))
                    data_b_channels.append(np.zeros(image_size, dtype=float))
                a = np.array(data_a_channels)
                a = np.transpose(a, (1, 2, 0))
                b = np.array(data_b_channels)
                b = np.transpose(b, (1, 2, 0))
                batch_data_a.append(a)
                batch_data_b.append(b)
                if len(batch_data_a) == batch_size:
                    yield np.array(batch_data_a), np.array(batch_data_b)
                    batch_data_a = list()
                    batch_data_b = list()


if __name__ == '__main__':
    train_generator = gan_data_generator('/home/yf/datas/NF/T12STIR_train.yaml', 8, (1152, 384), 3)
    data = next(train_generator)
