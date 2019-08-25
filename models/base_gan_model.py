from abc import ABC
from pathlib import Path
from models import BaseModel, get_model_class_by_name
from utils import yaml_utils


class BaseGanModel(BaseModel, ABC):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, **kwargs)
        self.kwargs = self._get_extend_kwargs(self.kwargs)
        self.generator_class = get_model_class_by_name(self.kwargs['model']['generator']['name'])
        self.discriminator_class = get_model_class_by_name(self.kwargs['model']['discriminator']['name'])
        self.G_channels = self.kwargs['model']['generator']['channels']
        self.D_channels = self.kwargs['model']['discriminator']['channels']

    @staticmethod
    def _get_extend_kwargs(kwargs):
        generator_config = 'configs/models/generator/' + kwargs['model']['generator']['name'] + '.yaml'
        if Path(generator_config).exists():
            generator_config = yaml_utils.read(generator_config)
            if generator_config is not None:
                kwargs['model']['generator'].update(generator_config)

        discriminator_config = 'configs/models/discriminator/' + kwargs['model']['discriminator']['name'] + '.yaml'
        if Path(discriminator_config).exists():
            discriminator_config = yaml_utils.read(discriminator_config)
            if discriminator_config is not None:
                kwargs['model']['discriminator'].update(discriminator_config)
        return kwargs
