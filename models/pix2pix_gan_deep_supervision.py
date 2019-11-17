import numpy as np
import tensorflow as tf

from models.base_gan_model import BaseGanModel
from models.pix2pix_gan import Pix2PixGAN
from models.utils.loss_funcation import l1_loss


class Pix2PixGANDeepSupervision(Pix2PixGAN):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self._lambda = self.kwargs['model']['lambda']
        self.build_model()
        self.summary()
        self.train_saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def build_model(self):
        # train generator
        data_shape = self.data_shape
        self.real_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], name='real_a')
        self.real_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels], name='real_b')

        self.fake_b_sample = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels],
                                            name='fake_b')
        self.fake_a_sample = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels],
                                            name='fake_a')
        # update Generator a2b
        self.fake_b = self.generator(self.real_a, is_training=True, name='generator_a2b')
        fake_logit_b = self.discriminator(self.fake_b[0], name='discriminator_b')
        self.g_loss_a2b = self.loss_fn(fake_logit_b, tf.ones_like(fake_logit_b)) + \
                          self._lambda * l1_loss(self.fake_b[0], self.real_b)
        # update Generator b2a
        self.fake_a = self.generator(self.real_b, is_training=True, name='generator_b2a')
        fake_logit_a = self.discriminator(self.fake_a[0], name='discriminator_a')
        self.g_loss_b2a = self.loss_fn(fake_logit_a, tf.ones_like(fake_logit_a)) + \
                          self._lambda * l1_loss(self.fake_a[0], self.real_a)

        # train discriminator
        real_logit_b = self.discriminator(self.real_b, reuse=True, name='discriminator_b')
        fake_logit_b = self.discriminator(self.fake_b_sample, reuse=True, name='discriminator_b')
        d_loss_real_b = self.loss_fn(real_logit_b, tf.ones_like(real_logit_b))
        d_loss_fake_b = self.loss_fn(fake_logit_b, tf.zeros_like(fake_logit_b))
        self.d_loss_b = d_loss_real_b + d_loss_fake_b

        real_logit_a = self.discriminator(self.real_a, reuse=True, name='discriminator_a')
        fake_logit_a = self.discriminator(self.fake_a_sample, reuse=True, name='discriminator_a')
        d_loss_real_a = self.loss_fn(real_logit_a, tf.ones_like(real_logit_a))
        d_loss_fake_a = self.loss_fn(fake_logit_a, tf.zeros_like(fake_logit_a))
        self.d_loss_a = d_loss_real_a + d_loss_fake_a

        # update Generator fakea2b
        fake_b2a2b = self.generator(self.fake_a_sample, is_training=True, reuse=True, name='generator_a2b')
        fake_b = self.generator(self.real_a, is_training=True, reuse=True, name='generator_a2b')
        deep_loss = 0
        for i in range(len(fake_b)):  # 中间全部再进行一次深度监督
            deep_loss += l1_loss(fake_b2a2b[i], fake_b[i])
        self.deep_loss = deep_loss

        train_vars = tf.trainable_variables()
        self.g_vars_a2b = [var for var in train_vars if 'generator_a2b' in var.name]
        self.d_vars_b = [var for var in train_vars if 'discriminator_b' in var.name]
        self.g_vars_b2a = [var for var in train_vars if 'generator_b2a' in var.name]
        self.d_vars_a = [var for var in train_vars if 'discriminator_a' in var.name]

        # eval or test
        self.test_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], name='test_a')
        self.test_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels], name='test_b')
        self.test_fake_b = self.generator(self.test_a, reuse=True, is_training=False, name='generator_a2b')
        self.test_metric_a2b = tf.reduce_mean(tf.image.ssim(self.test_fake_b[0], self.test_b, max_val=1.0))

    def summary(self):
        value_max = tf.reduce_max(self.real_a)
        real_a = self.real_a[:, :, :, self.in_channels // 2:self.in_channels - self.in_channels // 2] / value_max
        real_a_summary = tf.summary.image('{}/AReal'.format(self.dataset_name), real_a, max_outputs=1)

        value_max = tf.reduce_max(self.fake_b[0])
        fake_b = self.fake_b[0][:, :, :, self.out_channels // 2:self.out_channels - self.out_channels // 2] / value_max
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

    def train(self):
        """Train pix2pix"""
        g_optimizer_a2b = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.g_loss_a2b,
                                                                                     var_list=self.g_vars_a2b)
        g_optimizer_b2a = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.g_loss_b2a,
                                                                                     var_list=self.g_vars_b2a)
        deep_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.deep_loss,
                                                                                    var_list=self.g_vars_a2b)
        d_optimizer_b = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.d_loss_b,
                                                                                   var_list=self.d_vars_b)
        d_optimizer_a = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.d_loss_a,
                                                                                   var_list=self.d_vars_a)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.load(self.checkpoint_dir / 'train', self.train_saver)
        writer = tf.summary.FileWriter('../tensorboard_logs/{}/{}/{}'.format(self.dataset_name, self.name, self.tag),
                                       self.sess.graph)

        train_generator = self.train_data_loader.get_data_generator()
        train_size = self.train_data_loader.get_size()
        best_eval_metric = float('inf')
        for epoch in range(self.pre_epoch, self.total_epoch):
            lr = self.scheduler_fn(epoch)
            g_loss_sum = d_loss_sum = 0
            best_g_loss = float('inf')
            best_real_a = best_fake_b = best_real_b = np.zeros(
                shape=(self.batch_size, self.data_shape[0], self.data_shape[1], 1))
            for step in range(train_size):
                _, _, batch_a, _, _, batch_b = next(train_generator)
                # Update G network and record fake outputs
                fake_b, _, g_loss_a2b = self.sess.run([self.fake_b, g_optimizer_a2b, self.g_loss_a2b],
                                                      feed_dict={self.real_a: batch_a, self.real_b: batch_b,
                                                                 self.lr_tensor: lr})
                if best_g_loss >= g_loss_a2b:
                    best_g_loss, best_real_a, best_fake_b, best_real_b = (g_loss_a2b, batch_a, fake_b, batch_b)
                g_loss_sum += g_loss_a2b

                fake_a, _, g_loss_b2a = self.sess.run([self.fake_a, g_optimizer_b2a, self.g_loss_b2a],
                                                      feed_dict={self.real_a: batch_a, self.real_b: batch_b,
                                                                 self.lr_tensor: lr})
                # Update D network
                _, d_loss_b = self.sess.run([d_optimizer_b, self.d_loss_b],
                                            feed_dict={self.real_b: batch_b, self.fake_b_sample: fake_b[0],
                                                       self.lr_tensor: lr})
                d_loss_sum += d_loss_b
                _, d_loss_a = self.sess.run([d_optimizer_a, self.d_loss_a],
                                            feed_dict={self.real_a: batch_a, self.fake_a_sample: fake_a[0],
                                                       self.lr_tensor: lr})
                # Update G network with fake outputs
                _, g_loss_a2b2a = self.sess.run([g_optimizer_b2a, self.g_loss_b2a],
                                                feed_dict={self.real_a: batch_a, self.real_b: fake_b[0],
                                                           self.lr_tensor: lr})
                _, g_loss_b2a2b = self.sess.run([g_optimizer_a2b, self.g_loss_a2b],
                                                feed_dict={self.real_a: fake_a[0], self.real_b: batch_b,
                                                           self.lr_tensor: lr})

                _, deep_loss = self.sess.run([deep_optimizer, self.deep_loss],
                                             feed_dict={self.fake_a_sample: fake_a[0], self.real_a: batch_a,
                                                        self.lr_tensor: lr})

                print('{}/{} Epoch:{:>3d}/{:<3d} Step:{:>4d}/{:<4d} g_loss:{:<5.5f} d_loss:{:<5.5f}'
                      .format(self.name, self.tag, epoch, self.total_epoch, step, train_size, g_loss_a2b, d_loss_b))

            # eval G network
            eval_generator = self.eval_data_loader.get_data_generator()
            eval_size = self.eval_data_loader.get_size()
            eval_metric_sum = 0
            for step in range(eval_size):
                _, _, batch_a, _, _, batch_b = next(eval_generator)
                eval_metric = self.sess.run(self.test_metric_a2b,
                                            feed_dict={self.test_a: batch_a, self.test_b: batch_b})
                eval_metric_sum += eval_metric

            # draw summary
            image_summary = self.sess.run(self.image_summary,
                                          feed_dict={self.real_a: best_real_a, self.fake_b: best_fake_b,
                                                     self.real_b: best_real_b})
            scalar_summary = self.sess.run(self.scalar_summary,
                                           feed_dict={self.lr_tensor: lr, self.g_loss_a2b: g_loss_sum / train_size,
                                                      self.d_loss_b: d_loss_sum / train_size,
                                                      self.test_metric_a2b: eval_metric_sum / eval_size})
            writer.add_summary(image_summary, epoch)
            writer.add_summary(scalar_summary, epoch)

            # save model
            if best_eval_metric >= eval_metric_sum:
                self.save(self.checkpoint_dir / 'best', self.best_saver, epoch)
                best_eval_metric = eval_metric_sum
            if epoch % self.save_freq == 0:
                self.save(self.checkpoint_dir / 'train', self.train_saver, epoch)
