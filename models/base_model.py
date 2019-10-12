import tensorflow as tf
from abc import ABC, abstractmethod
from pathlib import Path
from models.utils.loss_funcation import get_loss_fn_by_name
from models.utils.metrics_funcation import get_metrics_fn_by_name
from models.utils.scheduler import get_scheduler_fn
from utils import yaml_utils
from utils.config_utils import dict_update


class BaseModel(ABC):
    def __init__(self, **kwargs):
        self.kwargs = BaseModel._get_extend_kwargs(kwargs)
        self.tag = kwargs['tag']
        self.sess = kwargs['sess']
        self.is_training = (kwargs['phase'] == 'train')
        self.print_freq = kwargs['print_freq']
        self.save_freq = kwargs['save_freq']
        self.lr_tensor = tf.placeholder(tf.float32, None, name='learning_rate')
        self.dataset_name = kwargs['dataset']['name']
        # model
        model = kwargs['model']
        self.name = model['name']
        self.epoch = model['epoch']
        self.batch_size = model['batch_size']
        self.in_channels = model['in_channels']
        self.out_channels = model['out_channels']
        self.filter_channels = model['filter_channels']
        self.loss_fn = get_loss_fn_by_name(model['loss'])
        self.metrics_fn = {metric: get_metrics_fn_by_name(metric) for metric in model['metrics']}
        self.scheduler_fn = get_scheduler_fn(total_epoch=self.epoch, **model['scheduler'])
        self.checkpoint_dir = Path(model['checkpoint_dir']) / self.dataset_name / self.name / self.tag

        self.train_data_loader = kwargs['train_data_loader']
        self.eval_data_loader = kwargs['eval_data_loader']
        self.test_data_loader = kwargs['test_data_loader']

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
            checkpoint_dir = Path(checkpoint_dir) / 'best'
        else:
            checkpoint_dir = Path(checkpoint_dir) / 'train'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saver.save(self.sess, str(checkpoint_dir / 'model'), global_step=epoch)

    def load(self, checkpoint_dir, is_best, **kwargs):
        if is_best:
            checkpoint_dir = Path(checkpoint_dir) / 'best'
        else:
            checkpoint_dir = Path(checkpoint_dir) / 'train'
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
