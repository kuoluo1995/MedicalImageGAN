import tensorflow as tf
from models.utils.layers import leaky_relu, instance_norm3d, conv3d


def build_model(image, filter_channels, reuse=False, name='patch_gan', **kwargs):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = leaky_relu(conv3d(image, filter_channels, name='d_h0_cov'))
        h1 = leaky_relu(instance_norm3d(conv3d(h0, filter_channels * 2, name='d_h1_conv'), 'd_bn1'))
        h2 = leaky_relu(instance_norm3d(conv3d(h1, filter_channels * 4, name='d_h2_conv'), 'd_bn2'))
        h3 = leaky_relu(instance_norm3d(conv3d(h2, filter_channels * 8, stride=1, name='d_h3_conv'), 'd_bn3'))
        h4 = conv3d(h3, 1, stride=1, name='d_h3_pred')
        return h4
