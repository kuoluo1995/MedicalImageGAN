from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy import ndimage

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
        image_size = self.image_size
        self.realA = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], self.in_channels], name='realA')
        self.realB = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], self.out_channels], name='realB')
        self.fakeB = self.generator(self.realA, name='generatorA2B')
        # self.metricB = {name: fn(self.fakeB, self.realB) for name, fn in self.metrics_fn.items()}

        fakeB_logit = self.discriminator(self.fakeB, name='discriminatorB')
        self.g_lossA2B = self.loss_fn(fakeB_logit, tf.ones_like(fakeB_logit)) + self._lambda * l1_loss(self.fakeB,
                                                                                                       self.realB)

        # train discriminator
        self.fakeB_sample = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], self.out_channels],
                                           name='fakeB')
        realB_logit = self.discriminator(self.realB, reuse=True, name='discriminatorB')
        fakeB_logit = self.discriminator(self.fakeB_sample, reuse=True, name='discriminatorB')

        self.d_loss_realB = self.loss_fn(realB_logit, tf.ones_like(realB_logit))
        self.d_loss_fakeB = self.loss_fn(fakeB_logit, tf.zeros_like(fakeB_logit))
        self.d_lossB = self.d_loss_realB + self.d_loss_fakeB

        train_vars = tf.trainable_variables()
        self.g_vars = [var for var in train_vars if 'generator' in var.name]
        self.d_vars = [var for var in train_vars if 'discriminator' in var.name]

        # eval
        self.testA = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], self.in_channels], name='testA')
        self.testB = tf.placeholder(tf.float32, [None, image_size[0], image_size[1], self.out_channels], name='testB')
        self.test_fakeB = self.generator(self.testA, reuse=True, name='generatorA2B')
        self.test_loss = l1_loss(self.test_fakeB, self.testB)
        # self.test_metric = {name: fn(self.test_fakeB, self.testB) for name, fn in self.metrics_fn.items()}

    def summary(self):
        self.lr_summary = tf.summary.scalar('{}/{}/LearningRate'.format(self.dataset_name, self.name), self.lr_tensor)

        value_max = tf.reduce_max(self.realA)
        realA = self.realA / value_max  # value_min must be 0
        realA_summary = tf.summary.image('{}/{}/AReal'.format(self.dataset_name, self.name),
                                         realA[:, :, :, self.in_channels // 2:self.in_channels - self.in_channels // 2],
                                         max_outputs=1)

        value_max = tf.reduce_max(self.fakeB)
        fakeB = self.fakeB / value_max
        tf.clip_by_value(fakeB, 0, 1)
        fakeB_summary = tf.summary.image('{}/{}/BFake'.format(self.dataset_name, self.name),
                                         fakeB[:, :, :,
                                         self.out_channels // 2:self.out_channels - self.out_channels // 2],
                                         max_outputs=1)

        value_max = tf.reduce_max(self.realB)
        realB = self.realB / value_max
        realB_summary = tf.summary.image('{}/{}/BReal'.format(self.dataset_name, self.name),
                                         realB[:, :, :,
                                         self.out_channels // 2:self.out_channels - self.out_channels // 2],
                                         max_outputs=1)
        self.g_image_summary = tf.summary.merge([realA_summary, realB_summary, fakeB_summary])

        # metric_sum = list()
        # for name, value in self.metricB.items():
        #     metric_sum.append(tf.summary.scalar('{}/{}/{}'.format(self.dataset_name, self.name, name), value))
        self.g_loss_A2B_summary = tf.summary.scalar('{}/{}/GLossA2B'.format(self.dataset_name, self.name),
                                                    self.g_lossA2B)
        # self.g_sum = tf.summary.merge([g_loss_A2B_sum])

        # d_loss_realB_sum = tf.summary.scalar('{}/{}/DLossRealB'.format(self.dataset_name, self.name), self.d_loss_realB)
        # d_loss_fakeB_sum = tf.summary.scalar('{}/{}/DLossFakeB'.format(self.dataset_name, self.name), self.d_loss_fakeB)
        self.d_loss_B_summary = tf.summary.scalar('{}/{}/DLossB'.format(self.dataset_name, self.name), self.d_lossB)
        # self.d_sum = tf.summary.merge([d_loss_B_sum])

        self.t_loss_summary = tf.summary.scalar('{}/{}/test_loss'.format(self.dataset_name, self.name), self.test_loss)
        # test_metric = list()
        # for name, value in self.test_metric.items():
        #     test_metric.append(tf.summary.scalar('{}/{}/test_{}'.format(self.dataset_name, self.name, name), value))
        # self.test_sum = tf.summary.merge([test_loss])

    def train(self):
        """Train cyclegan"""
        g_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.g_lossA2B, var_list=self.g_vars)
        d_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.d_lossB, var_list=self.d_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        # self.load(self.checkpoint_dir, is_best=True)
        writer = tf.summary.FileWriter('../tensorboard_logs/{}/{}/{}'.format(self.dataset_name, self.name, self.tag),
                                       self.sess.graph)
        train_generator = self.train_data_loader.get_data_generator()
        data_size = self.train_data_loader.get_size()

        best_eval_loss = float('inf')
        for epoch in range(self.epoch):
            lr = self.scheduler_fn(epoch)
            g_loss_sum = d_loss_sum = 0
            best_fakeB = best_realA = best_realB = np.zeros(
                shape=(self.batch_size, self.image_size[0], self.image_size[1], 1))
            best_g_loss = float('inf')
            for step in range(data_size):
                a_path, batchA, b_path, batchB = next(train_generator)

                # Update G network and record fake outputs
                fakeB, _, g_loss = self.sess.run([self.fakeB, g_optimizer, self.g_lossA2B],
                                                 feed_dict={self.realA: batchA, self.realB: batchB,
                                                            self.lr_tensor: lr})
                if best_g_loss > g_loss:
                    best_g_loss, best_fakeB, best_realA, best_realB = (g_loss, fakeB, batchA, batchB)
                g_loss_sum += g_loss
                # Update D network
                _, d_loss = self.sess.run([d_optimizer, self.d_lossB],
                                          feed_dict={self.realB: batchB, self.fakeB_sample: fakeB, self.lr_tensor: lr})
                d_loss_sum += d_loss
                print('Epoch:{:>3d}/{:<3d} Step:{:>4d}/{:<4d} g_loss:{:<5.5f} d_loss:{:<5.5f}'.format(epoch, self.epoch,
                                                                                                      step, data_size,
                                                                                                      g_loss, d_loss))

            # eval G network
            eval_generator = self.eval_data_loader.get_data_generator()
            eval_size = self.eval_data_loader.get_size()
            t_loss_sum = 0
            for step in range(eval_size):
                a_path, batchA, b_path, batchB = next(eval_generator)
                test_loss = self.sess.run([self.test_loss], feed_dict={self.testA: batchA, self.testB: batchB})
                t_loss_sum += test_loss

            # draw summary
            lr_summary = self.sess.run([self.lr_summary], feed_dict={self.lr_tensor: lr})
            g_summary = self.sess.run([self.g_loss_A2B_summary], feed_dict={self.g_lossA2B: g_loss_sum / data_size})
            d_summary = self.sess.run([self.d_loss_B_summary], feed_dict={self.d_lossB: d_loss_sum / data_size})
            image_summary = self.sess.run([self.g_image_summary],
                                          feed_dict={self.realA: best_realA, self.fakeB: best_fakeB,
                                                     self.realB: best_realB})
            t_loss_summary = self.sess.run([self.t_loss_summary], feed_dict={self.test_loss: t_loss_sum / eval_size})
            writer.add_summary([lr_summary, g_summary, d_summary, image_summary, t_loss_summary], epoch)

            if t_loss_sum <= best_eval_loss:
                self.save(self.checkpoint_dir, epoch, True)
                best_eval_loss = t_loss_sum
            if epoch % self.save_freq == 0:
                self.save(self.checkpoint_dir, epoch, False)

    def test(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.load(self.checkpoint_dir, is_best=True)
        data_generator = self.test_data_loader.get_data_generator()
        data_size = self.test_data_loader.get_size()
        pre_b_path = ''
        nii_model = list()
        sum_loss = 0
        for step in range(data_size):
            a_path, batchA, b_path, batchB = next(data_generator)
            fakeB, loss = self.sess.run([self.test_fakeB, self.test_loss],
                                        feed_dict={self.testA: batchA, self.testB: batchB})
            if pre_b_path != b_path:
                if pre_b_path != '':
                    b_nii_head = nii_header_reader(pre_b_path)
                    Path('./result/{}/{}/{}/'.format(self.dataset_name, self.name, self.tag)).mkdir(parents=True,
                                                                                                    exist_ok=True)
                    # resize images
                    nii_model = resize_data(np.array(nii_model), b_nii_head['header'].get_data_shape())
                    nii_writer(
                        'result/{}/{}/{}/fake_{}.nii'.format(self.dataset_name, self.name, self.tag,
                                                             Path(pre_b_path).parent.stem), b_nii_head,
                        np.array(nii_model))
                    print('Path:{} loss:{}'.format(pre_b_path, sum_loss))
                    nii_model = list()
                    sum_loss = 0
                pre_b_path = b_path
            nii_model.append(fakeB[0, :, :, self.out_channels // 2])
            sum_loss += loss
        if len(nii_model) > 0:
            b_nii_head = nii_header_reader(pre_b_path)
            nii_model = resize_data(np.array(nii_model), b_nii_head['header'].get_data_shape())
            nii_writer('result/{}/{}/{}/fake_{}.nii'.format(self.dataset_name, self.name, self.tag,
                                                            Path(pre_b_path).parent.stem), b_nii_head,
                       np.array(nii_model))
            print('Path:{} loss:{}'.format(pre_b_path, sum_loss))


def resize_data(data_, data_shape):  # resize for nf dataset
    data_ = np.transpose(data_, (1, 2, 0))
    shape = data_.shape
    d = shape[0]
    d_scale = 1.0
    if shape[0] <= data_shape[0]:
        d = data_shape[0]
    else:
        d_scale = shape[0] * 1.0 / data_shape[0]

    h = shape[1]
    h_scale = 1.0
    if shape[1] <= data_shape[1]:
        h = data_shape[1]
    else:
        h_scale = shape[1] * 1.0 / data_shape[1]

    data_ = ndimage.interpolation.zoom(data_, (d_scale, h_scale, 1.0), order=0)
    data_ = data_[:d, :h, :]
    return data_
