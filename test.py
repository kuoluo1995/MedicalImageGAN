import os
import tensorflow as tf
from models import get_model_class_by_name
from utils.config_utils import get_config


def _test(args):
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model_class = get_model_class_by_name(args['model']['name'])
        model = model_class(sess=sess, **args)
        model.test()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = get_config('base_pix')
    config['phase'] = 'test'
    _test(config)