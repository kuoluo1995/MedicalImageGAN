import importlib
from data_loader.base_data_generator import BaseDataGenerator


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
            'In {}.py, there should be a subclass of BaseDataGenerator with class name that matches {} in lowercase.'
                .format(name, target_cls))

    return target_cls
