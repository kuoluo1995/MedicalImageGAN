import tensorflow as tf
from models import BaseModel
from models.layers import conv2d, instance_norm, leaky_relu, deconv2d


class UNet(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, **kwargs)

    def build_model(self, image, reuse=False, name='unet', **kwargs):
        dropout_rate = 0.5 if self.is_training else 1.0
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            e1 = instance_norm(conv2d(image, self.filter_channels, name='g_e1_conv'))

            e2 = instance_norm(conv2d(leaky_relu(e1), self.filter_channels * 2, name='g_e2_conv'), 'g_bn_e2')

            e3 = instance_norm(conv2d(leaky_relu(e2), self.filter_channels * 4, name='g_e3_conv'), 'g_bn_e3')

            e4 = instance_norm(conv2d(leaky_relu(e3), self.filter_channels * 8, name='g_e4_conv'), 'g_bn_e4')

            e5 = instance_norm(conv2d(leaky_relu(e4), self.filter_channels * 8, name='g_e5_conv'), 'g_bn_e5')

            e6 = instance_norm(conv2d(leaky_relu(e5), self.filter_channels * 8, name='g_e6_conv'), 'g_bn_e6')

            e7 = instance_norm(conv2d(leaky_relu(e6), self.filter_channels * 8, name='g_e7_conv'), 'g_bn_e7')

            e8 = instance_norm(conv2d(leaky_relu(e7), self.filter_channels * 8, name='g_e8_conv'), 'g_bn_e8')

            d1 = deconv2d(tf.nn.relu(e8), self.filter_channels * 8, name='g_d1')
            d1 = tf.nn.dropout(d1, dropout_rate)
            d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)

            d2 = deconv2d(tf.nn.relu(d1), self.filter_channels * 8, name='g_d2')
            d2 = tf.nn.dropout(d2, dropout_rate)
            d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)

            d3 = deconv2d(tf.nn.relu(d2), self.filter_channels * 8, name='g_d3')
            d3 = tf.nn.dropout(d3, dropout_rate)
            d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)

            d4 = deconv2d(tf.nn.relu(d3), self.filter_channels * 8, name='g_d4')
            d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)

            d5 = deconv2d(tf.nn.relu(d4), self.filter_channels * 4, name='g_d5')
            d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)

            d6 = deconv2d(tf.nn.relu(d5), self.filter_channels * 2, name='g_d6')
            d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)

            d7 = deconv2d(tf.nn.relu(d6), self.filter_channels, name='g_d7')
            d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)

            d8 = deconv2d(tf.nn.relu(d7), self.out_channels, name='g_d8')

            return tf.nn.tanh(d8)
