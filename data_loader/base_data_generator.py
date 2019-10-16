from abc import abstractmethod


class BaseDataGenerator:
    def __init__(self, dataset_list, batch_size, image_size, in_channels, out_channels, is_training,
                 base_patch=(32, 32, 32)):
        self.dataset_list = dataset_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_training = is_training
        self.base_patch = base_patch

    @abstractmethod
    def get_size(self):
        pass

    @abstractmethod
    def get_data_generator(self):
        pass
