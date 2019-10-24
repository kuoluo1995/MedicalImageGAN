import os
import tensorflow as tf

from data_loader import get_data_loader_by_name
from models import get_model_class_by_name
from utils import yaml_utils
from utils.config_utils import get_config


def train(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # dataset
        dataset = args['dataset']
        model_dict = args['model']
        data_loader_class = get_data_loader_by_name(dataset['data_loader'])

        train_dict = yaml_utils.read(dataset['train_path'])
        train_data_loader = data_loader_class(True, **train_dict, **args)

        eval_dict = yaml_utils.read(dataset['eval_path'])
        eval_data_loader = data_loader_class(False, **eval_dict, **args)

        model_class = get_model_class_by_name(model_dict['name'])
        model = model_class(data_shape=train_data_loader.get_data_shape(), train_data_loader=train_data_loader,
                            eval_data_loader=eval_data_loader, test_data_loader=None, sess=sess, **args)
        model.train()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = get_config('base_2d_pix')
    config['tag'] = 'basic'
    # config['in_channels'] = 3
    # config['out_channels'] = 3
    train(config)
