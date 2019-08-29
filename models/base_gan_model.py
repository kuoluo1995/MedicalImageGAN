from abc import ABC
from models import BaseModel, get_model_fn


class BaseGanModel(BaseModel, ABC):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, **kwargs)
        self.kwargs = BaseGanModel._get_extend_kwargs(self.kwargs)
        model = self.kwargs['model']
        self.generator = get_model_fn('generator', out_channels=self.out_channels, is_training=self.is_training,
                                      **model['generator'])
        self.discriminator = get_model_fn('discriminator', **model['discriminator'])
