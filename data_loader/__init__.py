import importlib
from data_loader.base_data_generator import BaseDataGenerator


def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


def get_data_loader_by_name(name):
    file_name = 'data_loader.' + name
    libs = importlib.import_module(file_name)
    target_cls = name.replace('_', '')
    for key, cls in libs.__dict__.items():
        if target_cls.lower() == key.lower() and issubclass(cls, BaseDataGenerator):
            target_cls = cls
            break

    if issubclass(target_cls, str):
        raise NotImplementedError(
            'In {}.py, {} should be a subclass of BaseDataGenerator in lowercase.'.format(name, target_cls))
    return target_cls
