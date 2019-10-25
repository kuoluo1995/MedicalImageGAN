import tensorflow as tf
from models.utils.layers import leaky_relu, instance_norm3d, conv3d


def build_model(image, out_channels, filter_channels=64, reuse=False, name='patch_gan3d', is_training=False, **kwargs):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h1 = leaky_relu(conv3d(image, filter_channels, name='d_cov_h1'))
        h2 = leaky_relu(instance_norm3d(conv3d(h1, filter_channels * 2, name='d_conv_h2'), 'd_in1'))
        h3 = leaky_relu(instance_norm3d(conv3d(h2, filter_channels * 4, name='d_conv_h3'), 'd_in2'))
        h4 = leaky_relu(instance_norm3d(conv3d(h3, filter_channels * 8, stride=1, name='d_conv_h4'), 'd_in3'))
        h5 = conv3d(h4, out_channels, stride=1, name='d_pred_h5')
        return h5
