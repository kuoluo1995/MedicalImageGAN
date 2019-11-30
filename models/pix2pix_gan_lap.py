import numpy as np
import tensorflow as tf
from pathlib import Path

from models import get_model_fn
from models.base_gan_model import BaseGanModel
from models.pix2pix_gan import Pix2PixGAN
from models.utils.layers import down_sampling
from models.utils.loss_funcation import l1_loss
from utils import yaml_utils
from utils.nii_utils import nii_header_reader, nii_writer


class Pix2PixGANLap(Pix2PixGAN):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self.g_lambda = self.kwargs['model']['g_lambda']
        self.d_lambda = self.kwargs['model']['d_lambda']
        self.global_generator = get_model_fn('generator', out_channels=self.out_channels,
                                             **self.kwargs['model']['global_generator'])
        self.build_model()
        self.summary()
        self.train_saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def build_model(self):
        # train generator
        data_shape = self.data_shape
        self.real_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], name='real_a')
        self.real_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels], name='real_b')

        self.real_a_scales0 = self.real_a
        self.real_b_scales0 = self.real_b
        self.real_a_scales1 = down_sampling(self.real_a, 1)
        self.real_b_scales1 = down_sampling(self.real_b, 1)
        self.res0 = self.real_b_scales0 - self.real_a_scales0
        self.res1 = self.real_b_scales1 - self.real_a_scales1

        self.fake_res0 = self.generator(self.real_a_scales0, is_training=True, name='generator_res0')
        self.fake_res1 = self.global_generator(self.real_a_scales1, is_training=True, name='generator_res1')
        fake_ab0 = tf.concat([self.real_a_scales0, self.real_a_scales0 + self.fake_res0], 3)
        fake_logit_b0 = self.discriminator(fake_ab0, name='discriminator_b0')
        real_ab0 = tf.concat([self.real_a_scales0, self.real_b_scales0], 3)
        real_logit_b0 = self.discriminator(real_ab0, reuse=True, name='discriminator_b0')
        fake_ab1 = tf.concat([self.real_a_scales1, self.real_a_scales1 + self.fake_res1], 3)
        fake_logit_b1 = self.discriminator(fake_ab1, name='discriminator_b1')
        real_ab1 = tf.concat([self.real_a_scales1, self.real_b_scales1], 3)
        real_logit_b1 = self.discriminator(real_ab1, reuse=True, name='discriminator_b1')

        self.g_loss1 = self.loss_fn(fake_logit_b1, tf.ones_like(fake_logit_b1)) + self.g_lambda * l1_loss(
            self.fake_res1, self.res1)
        self.g_loss0 = self.loss_fn(fake_logit_b0, tf.ones_like(fake_logit_b0)) + \
                       self.g_lambda * l1_loss(self.fake_res0, self.res0) + l1_loss(self.fake_res1, self.res1)
        self.g_loss = self.g_loss1 + self.g_loss0

        # train discriminator
        d_real_loss1 = self.d_lambda * self.loss_fn(real_logit_b1, tf.ones_like(real_logit_b1)) + self.loss_fn(
            real_logit_b0, tf.ones_like(real_logit_b0))
        d_fake_loss1 = self.d_lambda * self.loss_fn(fake_logit_b1, tf.zeros_like(real_logit_b1)) + self.loss_fn(
            fake_logit_b0, tf.zeros_like(real_logit_b0))
        d_real_loss0 = self.d_lambda * self.loss_fn(real_logit_b0, tf.ones_like(real_logit_b0)) + self.loss_fn(
            real_logit_b0, tf.ones_like(real_logit_b0)) + self.loss_fn(real_logit_b1, tf.ones_like(real_logit_b1))
        d_fake_loss0 = self.d_lambda * self.loss_fn(real_logit_b0, tf.zeros_like(real_logit_b0)) + self.loss_fn(
            fake_logit_b0, tf.zeros_like(real_logit_b0)) + self.loss_fn(fake_logit_b1, tf.zeros_like(real_logit_b1))
        self.d_loss1 = d_real_loss1 + d_fake_loss1
        self.d_loss0 = d_real_loss0 + d_fake_loss0
        self.d_loss = self.d_loss1 + self.d_loss0

        train_vars = tf.trainable_variables()
        self.g_vars = [var for var in train_vars if 'generator' in var.name]
        self.d_vars = [var for var in train_vars if 'discriminator' in var.name]

        # eval or test
        self.test_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], name='test_a')
        self.test_res = self.generator(self.test_a, reuse=True, is_training=False, name='generator_res0')

    def summary(self):
        real_a0 = tf.summary.image('{}/0AReal'.format(self.dataset_name), self.real_a_scales0, max_outputs=1)
        real_b0 = tf.summary.image('{}/0BReal'.format(self.dataset_name), self.real_b_scales0, max_outputs=1)
        real_a1 = tf.summary.image('{}/1AReal'.format(self.dataset_name), self.real_a_scales1, max_outputs=1)
        real_b1 = tf.summary.image('{}/1BReal'.format(self.dataset_name), self.real_b_scales1, max_outputs=1)
        res0 = tf.summary.image('{}/0Res'.format(self.dataset_name), self.real_b_scales0 - self.real_a_scales0,
                                max_outputs=1)
        res1 = tf.summary.image('{}/1Res'.format(self.dataset_name), self.real_b_scales1 - self.real_a_scales1,
                                max_outputs=1)
        fake_res0 = tf.summary.image('{}/0FakeRes'.format(self.dataset_name), self.fake_res0, max_outputs=1)
        fake_res1 = tf.summary.image('{}/1FakeRes'.format(self.dataset_name), self.fake_res1, max_outputs=1)
        fake_b0 = tf.summary.image('{}/0BFake'.format(self.dataset_name), self.real_a_scales0 + self.fake_res0,
                                   max_outputs=1)
        fake_b1 = tf.summary.image('{}/1BFake'.format(self.dataset_name), self.real_a_scales1 + self.fake_res1,
                                   max_outputs=1)

        self.image_summary = tf.summary.merge(
            [real_a0, real_a1, real_b0, real_b1, res0, res1, fake_res0, fake_res1, fake_b0, fake_b1])

        lr_summary = tf.summary.scalar('{}/LearningRate'.format(self.dataset_name), self.lr_tensor)
        g_loss1 = tf.summary.scalar('{}/GLoss1'.format(self.dataset_name), self.g_loss1)
        g_loss0 = tf.summary.scalar('{}/GLoss0'.format(self.dataset_name), self.g_loss0)
        g_loss = tf.summary.scalar('{}/GLoss'.format(self.dataset_name), self.g_loss1 + self.g_loss0)
        d_loss1 = tf.summary.scalar('{}/DLoss1'.format(self.dataset_name), self.d_loss1)
        d_loss0 = tf.summary.scalar('{}/DLoss0'.format(self.dataset_name), self.d_loss0)
        d_loss = tf.summary.scalar('{}/DLoss'.format(self.dataset_name), self.d_loss1 + self.d_loss0)
        self.scalar_metric = tf.placeholder(tf.float32, None, name='metric')
        eval_metric_summary = tf.summary.scalar('{}/MetricA2B'.format(self.dataset_name), self.scalar_metric)
        self.scalar_summary = tf.summary.merge(
            [lr_summary, g_loss1, g_loss0, g_loss, d_loss1, d_loss0, d_loss, eval_metric_summary])

    def train(self):
        """Train pix2pix"""
        g_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        d_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # self.load(self.checkpoint_dir / self.test_model, self.train_saver)
        writer = tf.summary.FileWriter('../tensorboard_logs/{}/{}/{}'.format(self.dataset_name, self.name, self.tag),
                                       self.sess.graph)
        train_generator = self.train_data_loader.get_data_generator()
        train_size = self.train_data_loader.get_size()
        best_eval_metric = float('-inf')
        for epoch in range(self.pre_epoch, self.total_epoch):
            lr = self.scheduler_fn(epoch)
            g_loss1_sum = g_loss0_sum = d_loss1_sum = d_loss0_sum = 0  # sum one epoch g_loss and d_loss
            best_g_loss0 = float('inf')  # request min g_loss
            best_real_a0 = best_real_a1 = best_real_b0 = best_real_b1 = best_res0 = best_res1 = \
                np.zeros(shape=(self.batch_size, self.data_shape[0], self.data_shape[1], 1))
            for step in range(train_size):
                _, _, batch_a, _, _, batch_b = next(train_generator)
                # Update G network and record fake outputs
                real_a1, real_b1, fake_res0, fake_res1, _, g_loss1, g_loss0, d_loss1, d_loss0 = self.sess.run(
                    [self.real_a_scales1, self.real_b_scales1, self.fake_res0, self.fake_res1, g_optimizer,
                     self.g_loss1, self.g_loss0, self.d_loss1, self.d_loss0],
                    feed_dict={self.real_a: batch_a, self.real_b: batch_b, self.lr_tensor: lr})
                if best_g_loss0 >= g_loss0:  # min g_loss to show image
                    best_real_a0, best_real_a1, best_real_b0, best_real_b1, best_res0, best_res1, best_g_loss1, best_g_loss0, best_d_loss1, best_d_loss0 = (
                        batch_a, real_a1, batch_b, real_b1, fake_res0, fake_res1, g_loss1, g_loss0, d_loss1, d_loss0)
                g_loss0_sum += g_loss0
                g_loss1_sum += g_loss1
                # Update D network
                _, d_loss1, d_loss0 = self.sess.run([d_optimizer, self.d_loss1, self.d_loss0],
                                                    feed_dict={self.real_a: batch_a, self.real_b: batch_b,
                                                               self.lr_tensor: lr})
                d_loss1_sum += d_loss1
                d_loss0_sum += d_loss0
                print('{}/{} Epoch:{:>3d}/{:<3d} Step:{:>4d}/{:<4d} g_loss:{:<5.5f} d_loss:{:<5.5f}'
                      .format(self.name, self.tag, epoch, self.total_epoch, step, train_size, g_loss1 + g_loss0,
                              d_loss1 + d_loss0))
            # eval G network
            eval_generator = self.eval_data_loader.get_data_generator()
            eval_size = self.eval_data_loader.get_size()
            eval_metric_sum = 0
            num_eval_nii = 0
            current_path_b = ''
            nii_b = list()
            fake_nii_b = list()
            for step in range(eval_size + 1):
                path_a, _, batch_a, path_b, _, batch_b = next(eval_generator)
                if current_path_b != path_b:
                    if step > 0:
                        metrics = {name: fn(np.array(fake_nii_b), np.array(nii_b)) for name, fn in
                                   self.metrics_fn.items()}
                        eval_metric_sum += float(metrics['ssim_metrics'])
                        num_eval_nii += 1
                        nii_b = list()
                        fake_nii_b = list()
                    current_path_b = path_b
                    if step == eval_size:  # finnish eval
                        break
                fake_res = self.sess.run(self.test_res, feed_dict={self.test_a: batch_a})
                nii_b.append(batch_b[0, :, :, self.out_channels // 2])
                fake_nii_b.append(batch_a[0, :, :, self.in_channels // 2] + fake_res[0, :, :, self.out_channels // 2])

            # draw summary
            image_summary = self.sess.run(self.image_summary,
                                          feed_dict={self.real_a_scales0: best_real_a0,
                                                     self.real_b_scales0: best_real_b0,
                                                     self.real_a_scales1: best_real_a1,
                                                     self.real_b_scales1: best_real_b1, self.fake_res0: best_res0,
                                                     self.fake_res1: best_res1})
            scalar_summary = self.sess.run(self.scalar_summary,
                                           feed_dict={self.lr_tensor: lr, self.g_loss1: g_loss1_sum / train_size,
                                                      self.g_loss0: g_loss0_sum / train_size,
                                                      self.d_loss1: d_loss1_sum / train_size,
                                                      self.d_loss0: d_loss0_sum / train_size,
                                                      self.scalar_metric: eval_metric_sum / num_eval_nii})
            writer.add_summary(image_summary, epoch)
            writer.add_summary(scalar_summary, epoch)

            # save model
            if best_eval_metric <= eval_metric_sum:
                self.save(self.checkpoint_dir / 'best', self.best_saver, epoch)
                best_eval_metric = eval_metric_sum
            if epoch % self.save_freq == 0:
                self.save(self.checkpoint_dir / 'train', self.train_saver, epoch)

    def test(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.load(self.checkpoint_dir / self.test_model, self.best_saver)
        test_generator = self.test_data_loader.get_data_generator()
        test_size = self.test_data_loader.get_size()
        result_info = list()
        current_path_b = ''
        nii_b = list()
        fake_nii_b = list()
        for step in range(test_size + 1):
            path_a, source_path_a, batch_a, path_b, source_path_b, batch_b = next(test_generator)
            if current_path_b != path_b:
                if step > 0:
                    info = self._save_test_result(current_path_b, np.array(fake_nii_b), np.array(nii_b))
                    result_info.append(info)
                    nii_b = list()
                    fake_nii_b = list()
                current_path_b = path_b
                if step == test_size:
                    break
            fake_res = self.sess.run(self.test_res, feed_dict={self.test_a: batch_a})
            nii_b.append(batch_b[0, :, :, self.out_channels // 2])
            fake_nii_b.append(batch_a[0, :, :, self.in_channels // 2] + fake_res[0, :, :, self.out_channels // 2])
        yaml_utils.write('result/{}/{}/{}/{}/info.yaml'.format(self.dataset_name, self.name, self.tag, self.test_model),
                         result_info)

    def _save_test_result(self, current_path_b, fake_nii_b, nii_b):
        fake_nii_b = np.transpose(fake_nii_b, (1, 2, 0))
        nii_b = np.transpose(nii_b, (1, 2, 0))
        metrics = {name: fn(fake_nii_b, nii_b) for name, fn in self.metrics_fn.items()}
        metrics_str = ''
        for name, value in metrics.items():
            metrics_str += name + ':' + str(value) + ' '
        nii_head_b = nii_header_reader(current_path_b)
        nii_writer('result/{}/{}/{}/{}/fake_{}.nii'.format(self.dataset_name, self.name, self.tag, self.test_model,
                                                           Path(current_path_b).parent.stem), nii_head_b, fake_nii_b)
        print('Path:{} {}'.format(Path(current_path_b).parent.stem, metrics_str))
        info = {'index': Path(current_path_b).parent.stem}
        info.update({str(k): float(v) for k, v in metrics.items()})
        return info
