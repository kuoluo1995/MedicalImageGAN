import numpy as np
import tensorflow as tf
from scipy import ndimage
from pathlib import Path

from models.base_gan_model import BaseGanModel
from models.utils.loss_funcation import l1_loss
from utils.nii_utils import nii_header_reader, nii_writer


class Pix2PixGAN(BaseGanModel):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self._lambda = self.kwargs['model']['lambda']
        self.build_model()
        self.summary()
        self.saver = tf.train.Saver()

    def build_model(self):
        # train generator
        data_shape = self.data_shape
        self.real_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], name='real_a')
        self.real_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels], name='real_b')
        self.fake_b = self.generator(self.real_a, is_training=True, name='generator_a2b')

        fake_logit_b = self.discriminator(self.fake_b, name='discriminator_b')
        self.g_loss_a2b = self.loss_fn(fake_logit_b, tf.ones_like(fake_logit_b)) + self._lambda * l1_loss(self.fake_b,
                                                                                                          self.real_b)

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
        self.test_loss_a2b = l1_loss(self.test_fake_b, self.test_b)
        # self.test_metric = {name: fn(self.test_fake_b, self.test_b) for name, fn in self.metrics_fn.items()}

    def summary(self):
        value_max = tf.reduce_max(self.real_a)
        real_a = self.real_a[:, :, :, self.in_channels // 2:self.in_channels - self.in_channels // 2] / value_max
        real_a_summary = tf.summary.image('{}/{}/AReal'.format(self.dataset_name, self.name), real_a, max_outputs=1)

        value_max = tf.reduce_max(self.fake_b)
        fake_b = self.fake_b[:, :, :, self.out_channels // 2:self.out_channels - self.out_channels // 2] / value_max
        tf.clip_by_value(fake_b, 0, 1)
        fake_b_summary = tf.summary.image('{}/{}/BFake'.format(self.dataset_name, self.name), fake_b, max_outputs=1)

        value_max = tf.reduce_max(self.real_b)
        real_b = self.real_b[:, :, :, self.out_channels // 2:self.out_channels - self.out_channels // 2] / value_max
        real_b_summary = tf.summary.image('{}/{}/BReal'.format(self.dataset_name, self.name), real_b, max_outputs=1)
        self.g_image_summary = tf.summary.merge([real_a_summary, real_b_summary, fake_b_summary])

        lr_summary = tf.summary.scalar('{}/{}/LearningRate'.format(self.dataset_name, self.name), self.lr_tensor)
        # metric_sum = list()
        # for name, value in self.metricB.items():
        #     metric_sum.append(tf.summary.scalar('{}/{}/{}'.format(self.dataset_name, self.name, name), value))
        g_loss_a2b_summary = tf.summary.scalar('{}/{}/GLossA2B'.format(self.dataset_name, self.name), self.g_loss_a2b)
        d_loss_b_summary = tf.summary.scalar('{}/{}/DLossB'.format(self.dataset_name, self.name), self.d_loss_b)
        test_loss_a2b_summary = tf.summary.scalar('{}/{}/TestLossA2B'.format(self.dataset_name, self.name),
                                                  self.test_loss_a2b)
        self.scalar_summary = tf.summary.merge(
            [lr_summary, g_loss_a2b_summary, d_loss_b_summary, test_loss_a2b_summary])

    def train(self):
        """Train pix2pix"""
        g_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.g_loss_a2b, var_list=self.g_vars)
        d_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.d_loss_b, var_list=self.d_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        writer = tf.summary.FileWriter('../tensorboard_logs/{}/{}/{}'.format(self.dataset_name, self.name, self.tag),
                                       self.sess.graph)
        train_generator = self.train_data_loader.get_data_generator()
        train_size = self.train_data_loader.get_size()

        best_eval_loss = float('inf')
        for epoch in range(self.total_epoch):
            lr = self.scheduler_fn(epoch)
            g_loss_sum = d_loss_sum = 0
            best_g_loss = float('inf')
            best_real_a = best_fake_b = best_real_b = np.zeros(
                shape=(self.batch_size, self.data_shape[0], self.data_shape[1], 1))
            for step in range(train_size):
                _, _, batch_a, _, _, batch_b = next(train_generator)
                # Update G network and record fake outputs
                fake_b, _, g_loss = self.sess.run([self.fake_b, g_optimizer, self.g_loss_a2b],
                                                  feed_dict={self.real_a: batch_a, self.real_b: batch_b,
                                                             self.lr_tensor: lr})
                if best_g_loss >= g_loss:
                    best_g_loss, best_real_a, best_fake_b, best_real_b = (g_loss, batch_a, fake_b, batch_b)
                g_loss_sum += g_loss
                # Update D network
                _, d_loss = self.sess.run([d_optimizer, self.d_loss_b],
                                          feed_dict={self.real_b: batch_b, self.fake_b_sample: fake_b,
                                                     self.lr_tensor: lr})
                d_loss_sum += d_loss
                print('Epoch:{:>3d}/{:<3d} Step:{:>4d}/{:<4d} g_loss:{:<5.5f} d_loss:{:<5.5f}'.format(epoch,
                                                                                                      self.total_epoch,
                                                                                                      step, train_size,
                                                                                                      g_loss, d_loss))
            # eval G network
            eval_generator = self.eval_data_loader.get_data_generator()
            eval_size = self.eval_data_loader.get_size()
            eval_loss_sum = 0
            for step in range(eval_size):
                _, _, batch_a, _, _, batch_b = next(eval_generator)
                eval_loss = self.sess.run(self.test_loss_a2b, feed_dict={self.test_a: batch_a, self.test_b: batch_b})
                eval_loss_sum += eval_loss

            # draw summary
            image_summary = self.sess.run(self.g_image_summary,
                                          feed_dict={self.real_a: best_real_a, self.fake_b: best_fake_b,
                                                     self.real_b: best_real_b})
            scalar_summary = self.sess.run(self.scalar_summary,
                                           feed_dict={self.lr_tensor: lr, self.g_loss_a2b: g_loss_sum / train_size,
                                                      self.d_loss_b: d_loss_sum / train_size,
                                                      self.test_loss_a2b: eval_loss_sum / eval_size})
            writer.add_summary(image_summary, epoch)
            writer.add_summary(scalar_summary, epoch)

            # save model
            if best_eval_loss >= eval_loss_sum:
                self.save(self.checkpoint_dir, epoch, True)
                best_eval_loss = eval_loss_sum
            if epoch % self.save_freq == 0:
                self.save(self.checkpoint_dir, epoch, False)

    def test(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.load(self.checkpoint_dir, is_best=True)
        test_generator = self.test_data_loader.get_data_generator()
        test_size = self.test_data_loader.get_size()

        current_source_path_b = ''
        test_sum_loss = 0
        nii_b = list()
        for step in range(test_size):
            path_a, source_path_a, batch_a, path_b, source_path_b, batch_b = next(test_generator)
            fake_b, test_loss = self.sess.run([self.test_fake_b, self.test_loss_a2b],
                                              feed_dict={self.test_a: batch_a, self.test_b: batch_b})
            if current_source_path_b != source_path_b:
                if step > 0:
                    nii_head_b = nii_header_reader(current_source_path_b)
                    # resize images
                    nii_b = resize_data(np.array(nii_b), nii_head_b['header'].get_data_shape())
                    nii_writer('result/{}/{}/{}/fake_{}.nii'.format(self.dataset_name, self.name, self.tag,
                                                                    Path(current_source_path_b).parent.stem),
                               nii_head_b, np.array(nii_b))
                    print('Path:{} loss:{}'.format(current_source_path_b, test_sum_loss))
                    test_sum_loss = 0
                    nii_b = list()
                current_source_path_b = source_path_b
            nii_b.append(fake_b[0, :, :, self.out_channels // 2])
            test_sum_loss += test_loss

        if len(nii_b) > 0:
            nii_head_b = nii_header_reader(current_source_path_b)
            nii_b = resize_data(np.array(nii_b), nii_head_b['header'].get_data_shape())
            nii_writer('result/{}/{}/{}/fake_{}.nii'.format(self.dataset_name, self.name, self.tag,
                                                            Path(current_source_path_b).parent.stem), nii_head_b,
                       np.array(nii_b))
            print('Path:{} loss:{}'.format(current_source_path_b, test_sum_loss))


def resize_data(data_, target_shape):  # resize for nf dataset
    data_ = np.transpose(data_, (1, 2, 0))
    source_shape = data_.shape
    d = source_shape[0]
    d_scale = 1.0
    if source_shape[0] <= target_shape[0]:
        d = target_shape[0]
    else:
        d_scale = source_shape[0] * 1.0 / target_shape[0]

    h = source_shape[1]
    h_scale = 1.0
    if source_shape[1] <= target_shape[1]:
        h = target_shape[1]
    else:
        h_scale = source_shape[1] * 1.0 / target_shape[1]

    data_ = ndimage.interpolation.zoom(data_, (d_scale, h_scale, 1.0), order=0)
    data_ = data_[:d, :h, :]
    return data_
