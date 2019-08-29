import tensorflow as tf
from abc import ABC, abstractmethod
from pathlib import Path
from data_loader import get_data_loader_by_name
from models.utils.loss_funcation import get_loss_fn_by_name
from models.utils.scheduler import get_scheduler_fn
from utils import yaml_utils
from utils.config_utils import dict_update


class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.kwargs = BaseModel._get_extend_kwargs(kwargs)
        self.sess = kwargs['sess']
        self.tag = kwargs['tag']
        self.is_training = (kwargs['phase'] == 'train')
        self.print_freq = kwargs['print_freq']
        self.save_freq = kwargs['save_freq']
        # dataset
        dataset = kwargs['dataset']
        self.dataset_name = dataset['name']
        self.data_loader = get_data_loader_by_name(dataset['data_loader'])
        self.image_size = dataset['image_size']
        self.train_dataset = yaml_utils.read(dataset['train_path'])
        self.train_size = len(self.train_dataset)
        self.test_dataset = yaml_utils.read(dataset['test_path'])
        self.test_size = len(self.test_dataset)
        # model
        model = kwargs['model']
        self.epoch = model['epoch']
        self.batch_size = model['batch_size']
        self.in_channels = model['in_channels']
        self.out_channels = model['out_channels']
        self.filter_channels = model['filter_channels']
        self.loss_fn = get_loss_fn_by_name(model['loss']['name'])
        self.scheduler_fn = get_scheduler_fn(model['scheduler'])
        self.checkpoint_dir = Path(model['checkpoint_dir']) / kwargs['tag']

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    def summary(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def save(self, checkpoint_dir, epoch, **kwargs):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saver.save(self.sess, str(checkpoint_dir / self.dataset_name + '.ckpt'), global_step=epoch)

    def load(self, checkpoint_dir, **kwargs):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt = tf.train.get_checkpoint_state(str(checkpoint_dir / self.dataset_name + '.ckpt'))
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = Path(ckpt.model_checkpoint_path).stem
            self.saver.restore(self.sess, str(checkpoint_dir / ckpt_name))
        else:
            print('Loading checkpoint failed')

    @staticmethod
    def _get_extend_kwargs(kwargs):
        extend_config = 'configs/models/' + kwargs['model']['name'] + '.yaml'
        if Path(extend_config).exists():
            extend_config = yaml_utils.read(extend_config)
            if extend_config is not None:
                kwargs['model'] = dict_update(kwargs['model'], extend_config)
        return kwargs
