from abc import abstractmethod


class BaseDataGenerator:
    def __init__(self, is_augmented, dataset_list, data_shape, batch_size, in_channels, out_channels):
        self.is_augmented = is_augmented
        self.dataset_list = dataset_list
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def get_data_shape(self):
        pass

    @abstractmethod
    def get_size(self):
        pass

    @abstractmethod
    def get_data_generator(self):
        pass
