import tensorflow as tf
from models.utils.layers import leaky_relu, instance_norm3d, discrim_conv3d


def build_model(batch_input, out_channels, filter_channels=64, reuse=False, name='patch_gan3d', is_training=False,
                **kwargs):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h1 = leaky_relu(discrim_conv3d(batch_input, filter_channels, strides=(2, 2, 2), name='d_cov_h1'))
        h2 = leaky_relu(
            instance_norm3d(discrim_conv3d(h1, filter_channels * 2, strides=(2, 2, 2), name='d_conv_h2'), 'd_in1'))
        h3 = leaky_relu(
            instance_norm3d(discrim_conv3d(h2, filter_channels * 4, strides=(2, 2, 2), name='d_conv_h3'), 'd_in2'))
        h4 = discrim_conv3d(h3, out_channels, strides=(1, 1, 1), name='d_conv_h4')
        return h4
