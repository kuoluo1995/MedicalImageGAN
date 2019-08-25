import tensorflow as tf
from models import BaseModel
from models.layers import leaky_relu, conv2d, instance_norm


class PatchGan(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, **kwargs)

    def build_model(self, image, reuse=False, name='unet', **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            h0 = leaky_relu(conv2d(image, self.filter_channels, name='d_h0_cov'))
            h1 = leaky_relu(instance_norm(conv2d(h0, self.filter_channels * 2, name='d_h1_conv'), 'd_bn1'))
            h2 = leaky_relu(instance_norm(conv2d(h1, self.filter_channels * 4, name='d_h2_conv'), 'd_bn2'))
            h3 = leaky_relu(instance_norm(conv2d(h2, self.filter_channels * 8, stride=1, name='d_h3_conv'), 'd_bn3'))
            h4 = conv2d(h3, 1, stride=1, name='d_h3_pred')
            return h4
