import tensorflow as tf
from models.networks import leaky_relu, conv2d, instance_norm


def patch_gan(image, D_channels, reuse=False, name='discriminator', **kwargs):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        h0 = leaky_relu(conv2d(image, D_channels, name='d_h0_cov'))
        h1 = leaky_relu(instance_norm(conv2d(h0, D_channels * 2, name='d_h1_conv'), 'd_bn1'))
        h2 = leaky_relu(instance_norm(conv2d(h1, D_channels * 4, name='d_h2_conv'), 'd_bn2'))
        h3 = leaky_relu(instance_norm(conv2d(h2, D_channels * 8, stride=1, name='d_h3_conv'), 'd_bn3'))
        h4 = conv2d(h3, 1, stride=1, name='d_h3_pred')
        return h4
