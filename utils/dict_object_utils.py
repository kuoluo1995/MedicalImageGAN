from utils import yaml_utils
from pathlib import Path


def get_config(path):
    _dict = yaml_utils.read(path)
    _dict['name'] = Path(path).stem
    return _dict

def dict_update(_dict,extend_dict):
    for key, value in extend_dict.items():
        if isinstance(value, dict):
            _dict[key] = dict_update(_dict[key], extend_dict[key])
        else:
            _dict[key] = value
        return _dict

def object2dict(option):
    _dict = dict()
    for key, value in option.__dict__:
        if isinstance(value, Option):
            _dict[key] = object2dict(value)
        else:
            _dict[key] = value
    return _dict


def _dict2object(_dict):
    for key, value in _dict.items():
        if isinstance(value, dict):
            _dict[key] = Option(value)
        else:
            _dict[key] = value
    return _dict


class Option(dict):
    def __init__(self, *args, **kwargs):
        super(Option, self).__init__(*args, **kwargs)
        self.__dict__ = _dict2object(self)
