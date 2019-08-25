import tensorflow as tf
from models.networks import leaky_relu, conv2d, instance_norm


def patch_gan(image, D_channels, reuse=False, name='discriminator', **kwargs):
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = leaky_relu(conv2d(image, D_channels, name='d_h0_cov'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = leaky_relu(instance_norm(conv2d(h0, D_channels * 2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = leaky_relu(instance_norm(conv2d(h1, D_channels * 4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = leaky_relu(instance_norm(conv2d(h2, D_channels * 8, stride=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(h3, 1, stride=1, name='d_h3_pred')
        # h4 is (32 x 32 x 1)
        return h4
