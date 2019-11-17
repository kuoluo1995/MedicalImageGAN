import tensorflow as tf

from models.base_gan_model import BaseGanModel
from models.pix2pix_gan import Pix2PixGAN
from models.utils.loss_funcation import l1_loss


class Pix2PixGANTumor(Pix2PixGAN):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self._lambda = self.kwargs['model']['lambda']
        self._tumor_loss_threshold = self.kwargs['model']['tumor_loss_threshold']
        self._tumor_lambda = self.kwargs['model']['tumor_lambda']
        self.build_model()
        self.summary()
        self.train_saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def build_model(self):
        # train generator
        data_shape = self.data_shape
        self.real_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], name='real_a')
        self.real_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels], name='real_b')
        self._fake_b = self.generator(self.real_a, is_training=True, name='generator_a2b')
        real_b_detail = tf.clip_by_value(self.real_b, -1, self._tumor_loss_threshold)
        fake_b_detail = tf.clip_by_value(self._fake_b, -1, self._tumor_loss_threshold)
        # to improve detail
        zero = tf.zeros_like(self._fake_b)
        one = tf.ones_like(self._fake_b)
        fake_b_tumor = tf.where(self._fake_b > self._tumor_loss_threshold, x=one, y=zero)
        real_b_tumor = tf.where(self.real_b > self._tumor_loss_threshold, x=one, y=zero)
        # to improve tumor
        fake_ab = tf.concat([self.real_a, self._fake_b], 3)
        fake_logit_b = self.discriminator(fake_ab, name='discriminator_b')
        self.g_loss_a2b = self.loss_fn(fake_logit_b, tf.ones_like(fake_logit_b)) + self._lambda * l1_loss(
            fake_b_detail, real_b_detail) + self._tumor_lambda * l1_loss(fake_b_tumor, real_b_tumor)

        # train discriminator
        self.fake_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels], name='fake_b')
        real_ab = tf.concat([self.real_a, self.real_b], 3)
        fake_ab = tf.concat([self.real_a, self.fake_b], 3)
        real_logit_b = self.discriminator(real_ab, reuse=True, name='discriminator_b')
        fake_logit_b = self.discriminator(fake_ab, reuse=True, name='discriminator_b')
        d_loss_real_b = self.loss_fn(real_logit_b, tf.ones_like(real_logit_b))
        d_loss_fake_b = self.loss_fn(fake_logit_b, tf.zeros_like(fake_logit_b))
        self.d_loss_b = d_loss_real_b + d_loss_fake_b

        train_vars = tf.trainable_variables()
        self.g_vars = [var for var in train_vars if 'generator' in var.name]
        self.d_vars = [var for var in train_vars if 'discriminator' in var.name]

        # eval or test
        self.test_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], name='test_a')
        self.test_fake_b = self.generator(self.test_a, reuse=True, is_training=False, name='generator_a2b')
