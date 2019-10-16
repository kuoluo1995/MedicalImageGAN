import os
import tensorflow as tf

from data_loader import get_data_loader_by_name
from models import get_model_class_by_name
from utils import yaml_utils
from utils.config_utils import get_config


def _test(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        # dataset
        dataset = args['dataset']
        data_loader_class = get_data_loader_by_name(dataset['data_loader'])
        test_dict = yaml_utils.read(dataset['test_path'])
        test_data_loader = data_loader_class(test_dict['dataset'], args['model']['batch_size'], test_dict['shape'],
                                             args['model']['in_channels'], args['model']['out_channels'], False)
        model_class = get_model_class_by_name(args['model']['name'])
        model = model_class(image_size=test_data_loader.get_image_size(), train_data_loader=None, eval_data_loader=None,
                            test_data_loader=test_data_loader, sess=sess, **args)
        model.test()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = get_config('base_2d_pix')
    config['phase'] = 'test'
    config['tag'] = 'basic'
    _test(config)
