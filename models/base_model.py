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
        self.is_training = kwargs['is_training']
        self.test_model = kwargs['test_model']
        self.pre_epoch = kwargs['pre_epoch']
        self.total_epoch = kwargs['total_epoch']
        self.save_freq = kwargs['save_freq']
        self.batch_size = kwargs['batch_size']
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']

        self.dataset_name = kwargs['dataset']['name']
        self.data_shape = kwargs['data_shape']
        self.train_data_loader = kwargs['train_data_loader']
        self.eval_data_loader = kwargs['eval_data_loader']
        self.test_data_loader = kwargs['test_data_loader']

        # model
        model = kwargs['model']
        self.name = model['name']
        self.filter_channels = model['filter_channels']
        self.loss_fn = get_loss_fn_by_name(model['loss'])
        self.metrics_fn = {metric: get_metrics_fn_by_name(metric) for metric in model['metrics']}
        self.scheduler_fn = get_scheduler_fn(total_epoch=self.total_epoch, **model['scheduler'])
        self.lr_tensor = tf.placeholder(tf.float32, None, name='learning_rate')
        self.checkpoint_dir = Path(model['checkpoint_dir']) / self.dataset_name / self.name / self.tag

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

    def save(self, checkpoint_dir, saver, epoch, **kwargs):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        saver.save(self.sess, str(checkpoint_dir / 'model.cpk'), global_step=epoch)

    def load(self, checkpoint_dir, saver, **kwargs):
        # checkpoint = tf.train.get_checkpoint_state(str(checkpoint_dir))
        checkpoint = tf.train.latest_checkpoint(str(checkpoint_dir))
        if checkpoint:
            # saver.restore(self.sess, checkpoint.model_checkpoint_path)
            saver.restore(self.sess, checkpoint)
            self.pre_epoch = int(checkpoint.split('/')[-1].split('-')[-1])
        else:
            print('Loading checkpoint failed')
            self.pre_epoch = 0

    @staticmethod
    def _get_extend_kwargs(kwargs):
        config_path = 'configs/models/' + kwargs['model']['name'] + '.yaml'
        if Path(config_path).exists():
            extend_config = yaml_utils.read(config_path)
            if extend_config is not None:
                kwargs['model'] = dict_update(kwargs['model'], extend_config)
        return kwargs
