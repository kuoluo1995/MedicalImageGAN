import tensorflow as tf
from abc import ABC, abstractmethod
from pathlib import Path
from data_loader import get_data_generator_fn_by_name
from models.loss_funcation import get_loss_fn_by_name
from utils import yaml_utils


class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.kwargs = BaseModel._get_extend_kwargs(kwargs)
        if 'sess' in self.kwargs.keys():
            self.sess = self.kwargs['sess']
        self.name = self.kwargs['name']
        self.is_training = (self.kwargs['phase'] == 'train')
        self.print_freq = self.kwargs['print_freq']
        self.save_freq = self.kwargs['save_freq']
        # self.saver = tf.train.Saver()
        # dataset
        self.image_size = self.kwargs['dataset']['image_size']
        self.train_path = self.kwargs['dataset']['train_path']
        self.test_path = self.kwargs['dataset']['test_path']
        # model
        self.batch_size = self.kwargs['model']['batch_size']
        self.in_channels = self.kwargs['model']['in_channels']
        self.filter_channels = self.kwargs['model']['filter_channels']
        self.out_channels = self.kwargs['model']['out_channels']
        self.loss_fn = get_loss_fn_by_name(self.kwargs['model']['loss']['name'])
        self.epoch = self.kwargs['model']['epoch']
        self.epoch_step = self.kwargs['model']['epoch_step']
        self.lr = self.kwargs['model']['learning_rate']
        self.betal = self.kwargs['model']['betal']
        self.checkpoint_dir = Path(self.kwargs['model']['checkpoint_dir']) / self.kwargs['tag']
        self.sample_dir = Path(self.kwargs['model']['sample_dir']) / self.kwargs['tag']
        self.data_generator_fn = get_data_generator_fn_by_name(self.kwargs['dataset']['data_loader'])

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    def summary(self):
        pass

    def train(self):
        pass

    def save(self, checkpoint_dir, epoch, **kwargs):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saver.save(self.sess, str(checkpoint_dir / self.name + '.ckpt'), global_step=epoch)

    def sample_model(self, sample_dir, epoch, step, **kwargs):
        pass

    @staticmethod
    def _get_extend_kwargs(kwargs):
        extend_config = 'configs/models/' + kwargs['model']['name'] + '.yaml'
        if Path(extend_config).exists():
            extend_config = yaml_utils.read(extend_config)
            if extend_config is not None:
                kwargs['model'].update(extend_config)
        return kwargs
