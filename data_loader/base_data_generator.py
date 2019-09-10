from abc import abstractmethod


class BaseDataGenerator:
    def __init__(self, dataset, batch_size, image_size, channels, is_training):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.is_training = is_training

    @abstractmethod
    def get_size(self):
        pass

    @abstractmethod
    def get_data(self):
        pass
