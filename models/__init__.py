import importlib
from models.base_model import BaseModel


def get_model_class_by_name(name):
    file_name = 'models.' + name
    libs = importlib.import_module(file_name)
    target_cls = name.replace('_', '')
    for key, cls in libs.__dict__.items():
        if target_cls.lower() == key.lower() and issubclass(cls, BaseModel):
            target_cls = cls
            break

    if issubclass(target_cls, str):
        raise NotImplementedError(
            "In {}.py, there should be a subclass of BaseModel with class name that matches {} in lowercase.".format(
                name, target_cls))

    return target_cls
