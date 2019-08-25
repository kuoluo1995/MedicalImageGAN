import tensorflow as tf
import os
from configs.option import get_config
from models.cycle_gan import CycleGAN

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    config = get_config('configs/cycle_gan.yaml')
    config['tag'] = 'nf_cycle_gan'
    tf.set_random_seed(19)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = CycleGAN(sess, config)
        model.train()
