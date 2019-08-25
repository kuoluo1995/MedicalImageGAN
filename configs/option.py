from utils import yaml_utils


def get_config(path):
    _dict = yaml_utils.read(path)
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
