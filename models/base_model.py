import tensorflow as tf
from abc import ABC, abstractmethod
from pathlib import Path
from data_loader import get_data_loader_by_name
from models.utils.loss_funcation import get_loss_fn_by_name
from models.utils.metrics_funcation import get_metrics_fn_by_name
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
        self.lr_tensor = tf.placeholder(tf.float32, None, name='learning_rate')
        # dataset
        dataset = kwargs['dataset']
        self.dataset_name = dataset['name']
        data_loader_class = get_data_loader_by_name(dataset['data_loader'])
        self.image_size = tuple(dataset['image_size'])
        # model
        model = kwargs['model']
        self.name = model['name']
        self.epoch = model['epoch']
        self.batch_size = model['batch_size']
        self.in_channels = model['in_channels']
        self.out_channels = model['out_channels']
        self.filter_channels = model['filter_channels']
        self.loss_fn = get_loss_fn_by_name(model['loss'])
        self.metrics_fn = get_metrics_fn_by_name(model['metrics'])
        self.scheduler_fn = get_scheduler_fn(total_epoch=self.epoch, **model['scheduler'])
        self.checkpoint_dir = Path(model['checkpoint_dir']) / self.dataset_name / self.name

        self.train_data_loader = data_loader_class(yaml_utils.read(dataset['train_path']), self.batch_size,
                                                   self.image_size, self.in_channels, True)
        self.test_data_loader = data_loader_class(yaml_utils.read(dataset['test_path']), 1, self.image_size,
                                                  self.in_channels, False)

    @abstractmethod
    def build_model(self, **kwargs):
        pass

    def summary(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    def save(self, checkpoint_dir, epoch, is_best, **kwargs):
        if is_best:
            checkpoint_dir = Path(checkpoint_dir) / self.tag / 'best'
        else:
            checkpoint_dir = Path(checkpoint_dir) / self.tag / self.tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saver.save(self.sess, str(checkpoint_dir / 'model'), global_step=epoch)

    def load(self, checkpoint_dir, is_best, **kwargs):
        if is_best:
            checkpoint_dir = Path(checkpoint_dir) / self.tag / 'best'
        else:
            checkpoint_dir = Path(checkpoint_dir) / self.tag / self.tag
        ckpt = tf.train.get_checkpoint_state(str(checkpoint_dir))
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
