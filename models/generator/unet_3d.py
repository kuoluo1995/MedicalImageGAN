import tensorflow as tf
from models.utils.layers import conv3d, instance_norm, leaky_relu, deconv3d


def build_model(image, out_channels, filter_channels=64, reuse=False, name='3dunet', is_training=False, **kwargs):
    dropout_rate = 0.5 if is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        e1 = instance_norm(conv3d(image, filter_channels, name='g_e1_conv'))
        e2 = instance_norm(conv3d(leaky_relu(e1), filter_channels * 2, name='g_e2_conv'), 'g_bn_e2')
        e3 = instance_norm(conv3d(leaky_relu(e2), filter_channels * 4, name='g_e3_conv'), 'g_bn_e3')
        e4 = instance_norm(conv3d(leaky_relu(e3), filter_channels * 8, name='g_e4_conv'), 'g_bn_e4')
        e5 = instance_norm(conv3d(leaky_relu(e4), filter_channels * 8, name='g_e5_conv'), 'g_bn_e5')

        d1 = deconv3d(tf.nn.relu(e5), filter_channels * 8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e4], 1)

        d2 = deconv3d(tf.nn.relu(d1), filter_channels * 4, name='g_d2')
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e3], 1)

        d3 = deconv3d(tf.nn.relu(d2), filter_channels * 2, name='g_d3')
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e2], 1)

        d4 = deconv3d(tf.nn.relu(d3), filter_channels, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e1], 1)

        d8 = deconv3d(tf.nn.relu(d4), out_channels, name='g_d5')

        return tf.nn.tanh(d8)
