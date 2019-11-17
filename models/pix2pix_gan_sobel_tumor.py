import tensorflow as tf

from models.base_gan_model import BaseGanModel
from models.pix2pix_gan import Pix2PixGAN
from models.utils.loss_funcation import l1_loss


class Pix2PixGANSobelTumor(Pix2PixGAN):
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
        self.fake_b = self.generator(self.real_a, is_training=True, name='generator_a2b')
        fake_logit_b = self.discriminator(self.fake_b, name='discriminator_b')

        fake_b_detail = tf.clip_by_value(self.fake_b, 0, self._tumor_loss_threshold) / self._tumor_loss_threshold
        real_b_detail = tf.clip_by_value(self.real_b, 0, self._tumor_loss_threshold) / self._tumor_loss_threshold
        # 把肿瘤两极化来突出
        zero = tf.zeros_like(self.fake_b)
        one = tf.ones_like(self.fake_b)
        fake_b_tumor = tf.where(self.fake_b > self._tumor_loss_threshold, x=one, y=zero)
        real_b_tumor = tf.where(self.real_b > self._tumor_loss_threshold, x=one, y=zero)
        # 突出肿瘤
        self.g_loss_a2b = self.loss_fn(fake_logit_b, tf.ones_like(fake_logit_b)) + self._lambda * l1_loss(
            fake_b_detail, real_b_detail) + self._tumor_lambda * l1_loss(fake_b_tumor, real_b_tumor)

        # train discriminator
        self.fake_b_sample = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels],
                                            name='fake_b')
        real_logit_b = self.discriminator(self.real_b, reuse=True, name='discriminator_b')
        fake_logit_b = self.discriminator(self.fake_b_sample, reuse=True, name='discriminator_b')

        self.d_loss_real_b = self.loss_fn(real_logit_b, tf.ones_like(real_logit_b))
        self.d_loss_fake_b = self.loss_fn(fake_logit_b, tf.zeros_like(fake_logit_b))
        self.d_loss_b = self.d_loss_real_b + self.d_loss_fake_b

        train_vars = tf.trainable_variables()
        self.g_vars = [var for var in train_vars if 'generator' in var.name]
        self.d_vars = [var for var in train_vars if 'discriminator' in var.name]

        # eval or test
        self.test_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], name='test_a')
        self.test_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels], name='test_b')
        self.test_fake_b = self.generator(self.test_a, reuse=True, is_training=False, name='generator_a2b')
        self.test_metric_a2b = tf.reduce_mean(tf.image.ssim(self.test_fake_b, self.test_b, max_val=1.0))

    def summary(self):
        value_max = tf.reduce_max(self.real_a)
        real_a = self.real_a[:, :, :, 0:1] / value_max
        real_a_summary = tf.summary.image('{}/AReal'.format(self.dataset_name), real_a, max_outputs=1)

        value_max = tf.reduce_max(self.fake_b)
        fake_b = self.fake_b[:, :, :, self.out_channels // 2:self.out_channels - self.out_channels // 2] / value_max
        tf.clip_by_value(fake_b, 0, 1)
        fake_b_summary = tf.summary.image('{}/BFake'.format(self.dataset_name), fake_b, max_outputs=1)

        value_max = tf.reduce_max(self.real_b)
        real_b = self.real_b[:, :, :, self.out_channels // 2:self.out_channels - self.out_channels // 2] / value_max
        real_b_summary = tf.summary.image('{}/BReal'.format(self.dataset_name), real_b, max_outputs=1)
        self.image_summary = tf.summary.merge([real_a_summary, real_b_summary, fake_b_summary])

        lr_summary = tf.summary.scalar('{}/LearningRate'.format(self.dataset_name), self.lr_tensor)
        g_loss_a2b_summary = tf.summary.scalar('{}/GLossA2B'.format(self.dataset_name), self.g_loss_a2b)
        d_loss_b_summary = tf.summary.scalar('{}/DLossB'.format(self.dataset_name), self.d_loss_b)
        eval_metric_a2b_summary = tf.summary.scalar('{}/MetricA2B'.format(self.dataset_name), self.test_metric_a2b)
        self.scalar_summary = tf.summary.merge(
            [lr_summary, g_loss_a2b_summary, d_loss_b_summary, eval_metric_a2b_summary])
