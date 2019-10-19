import tensorflow as tf
from models.utils.layers import conv3d, instance_norm3d, leaky_relu, deconv3d


def build_model(image, out_channels, filter_channels=64, reuse=False, name='3dunet', is_training=False, **kwargs):
    dropout_rate = 0.5 if is_training else 1.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        e1 = instance_norm3d(conv3d(image, filter_channels, name='g_e1_conv'))
        e2 = instance_norm3d(conv3d(leaky_relu(e1), filter_channels * 2, name='g_e2_conv'), 'g_bn_e2')
        e3 = instance_norm3d(conv3d(leaky_relu(e2), filter_channels * 4, name='g_e3_conv'), 'g_bn_e3')
        e4 = instance_norm3d(conv3d(leaky_relu(e3), filter_channels * 8, name='g_e4_conv'), 'g_bn_e4')
        e5 = instance_norm3d(conv3d(leaky_relu(e4), filter_channels * 16, stride=(2, 2, 1), name='g_e5_conv'),
                             'g_bn_e5')
        e6 = instance_norm3d(conv3d(leaky_relu(e5), filter_channels * 16, stride=(2, 2, 1), name='g_e6_conv'),
                             'g_bn_e6')
        e7 = instance_norm3d(conv3d(leaky_relu(e6), filter_channels * 16, stride=(2, 1, 1), name='g_e7_conv'),
                             'g_bn_e7')
        e8 = instance_norm3d(conv3d(leaky_relu(e7), filter_channels * 32, stride=(1, 1, 1), name='g_e8_conv'),
                             'g_bn_e8')

        d1 = deconv3d(tf.nn.relu(e8), filter_channels * 16, stride=(1, 1, 1), name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm3d(d1, 'g_bn_d1'), e7], 4)

        d2 = deconv3d(tf.nn.relu(d1), filter_channels * 16, stride=(2, 1, 1), name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm3d(d2, 'g_bn_d2'), e6], 4)

        d3 = deconv3d(tf.nn.relu(d2), filter_channels * 16, stride=(2, 2, 1), name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm3d(d3, 'g_bn_d3'), e5], 4)

        d4 = deconv3d(tf.nn.relu(d3), filter_channels * 8, stride=(2, 2, 1), name='g_d4')
        d4 = tf.concat([instance_norm3d(d4, 'g_bn_d4'), e4], 4)

        d5 = deconv3d(tf.nn.relu(d4), filter_channels * 4, name='g_d5')
        d5 = tf.concat([instance_norm3d(d5, 'g_bn_d5'), e3], 4)

        d6 = deconv3d(tf.nn.relu(d5), filter_channels * 2, name='g_d6')
        d6 = tf.concat([instance_norm3d(d6, 'g_bn_d6'), e2], 4)

        d7 = deconv3d(tf.nn.relu(d6), filter_channels * 2, name='g_d7')
        d7 = tf.concat([instance_norm3d(d7, 'g_bn_d7'), e1], 4)

        d8 = deconv3d(tf.nn.relu(d7), out_channels, name='g_d8')

        return tf.nn.tanh(d8)
