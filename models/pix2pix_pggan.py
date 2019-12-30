import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.ndimage import zoom

from models.base_gan_model import BaseGanModel
from models.utils.loss_funcation import l1_loss
from utils import yaml_utils
from utils.nii_utils import nii_header_reader, nii_writer


class Pix2PixPGGAN(BaseGanModel):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self.model_id = self.kwargs['model_id']
        self.max_id = self.kwargs['max_id']
        self._lambda = self.kwargs['model']['lambda']
        self.output_size = self.kwargs['output_size']
        self.process_size = self.kwargs['process_size']
        self.is_transition = self.kwargs['is_transition']

        self.save_path = self.kwargs['save_path']
        self.read_path = self.kwargs['read_path']
        self.log_path = self.kwargs['log_path']

        self.build_model()

        self.write_saver = tf.train.Saver(self.dis_vars + self.gen_vars)
        self.read_saver = tf.train.Saver(self.dis_n_vars_read + self.gen_n_vars_read)
        if len(self.dis_out_vars_read + self.gen_out_vars_read):
            self.out_saver = tf.train.Saver(self.dis_out_vars_read + self.gen_out_vars_read)

        self.scales = list()
        self.images = list()
        self.summary()

    def build_model(self):
        # train generator
        self.real_a = tf.placeholder(tf.float32, [self.batch_size, *self.data_shape, self.in_channels], name='real_a')
        self.real_b = tf.placeholder(tf.float32, [self.batch_size, *self.data_shape, self.out_channels], name='real_b')
        self.keep_prob_tensor = tf.placeholder(tf.float32, name='keep_prob')
        self.epoch_tensor = tf.placeholder(tf.float32, shape=None, name='epoch')
        self.alpha_transition = tf.Variable(initial_value=0.0, trainable=False, name='alpha_transition')

        self.fake_b = self.generator(self.real_a, self.batch_size, self.process_size, self.is_transition,
                                     self.alpha_transition, self.keep_prob_tensor, name='generator_a2b')

        fake_ab = tf.concat([self.real_a, self.fake_b], 3)
        fake_logit_b = self.discriminator(fake_ab, self.process_size, self.is_transition, self.alpha_transition,
                                          name='discriminator_b')
        self.gen_loss_a2b = self.loss_fn(fake_logit_b, tf.ones_like(fake_logit_b)) + \
                            self._lambda * l1_loss(self.fake_b, self.real_b)

        # train discriminator
        real_ab = tf.concat([self.real_a, self.real_b], 3)
        fake_ab = tf.concat([self.real_a, self.fake_b], 3)
        real_logit_b = self.discriminator(real_ab, self.process_size, self.is_transition, self.alpha_transition,
                                          reuse=True, name='discriminator_b')
        fake_logit_b = self.discriminator(fake_ab, self.process_size, self.is_transition, self.alpha_transition,
                                          reuse=True, name='discriminator_b')
        dis_loss_real_b = self.loss_fn(real_logit_b, tf.ones_like(real_logit_b))
        dis_loss_fake_b = self.loss_fn(fake_logit_b, tf.zeros_like(fake_logit_b))
        # # gradient penalty from WGAN-GP todo read
        # differences = self.fake_b - self.real_b
        # self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.0)
        # interpolates_b = self.alpha * differences + self.real_b
        # interpolates_ab = tf.concat([self.real_a, interpolates_b], 3)
        # interpolates_logit = self.discriminator(interpolates_ab, self.process_size, self.is_transition,
        #                                         self.alpha_transition, reuse=True, name='discriminator')
        # gradients = tf.gradients(interpolates_logit, [interpolates_ab])[0]
        # # 2 norm
        # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        # gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
        self.dis_loss_b = dis_loss_real_b + dis_loss_fake_b

        # Training Parameters
        train_vars = tf.trainable_variables()
        self.gen_vars = [var for var in train_vars if 'gen' in var.name]
        self.dis_vars = [var for var in train_vars if 'dis' in var.name]

        # save the variables , which remain unchanged
        gen_n_vars = [var for var in self.gen_vars if 'gen_n' in var.name]
        dis_n_vars = [var for var in self.dis_vars if 'dis_n' in var.name]

        gen_out_vars = [var for var in self.gen_vars if 'gen_out' in var.name]
        dis_out_vars = [var for var in self.dis_vars if 'dis_out' in var.name]

        # read saved variables from previous model
        self.gen_n_vars_read = [var for var in gen_n_vars if str(self.output_size[0]) not in var.name]
        self.dis_n_vars_read = [var for var in dis_n_vars if str(self.output_size[0]) not in var.name]

        self.dis_out_vars_read = [var for var in gen_out_vars if str(self.output_size[0]) not in var.name]
        self.gen_out_vars_read = [var for var in dis_out_vars if str(self.output_size[0]) not in var.name]

    def summary(self):
        # Summary
        self.scales.append(tf.summary.scalar('{}/LearningRate'.format(self.dataset_name), self.lr_tensor))
        self.scales.append(tf.summary.scalar('{}/GLossA2B'.format(self.dataset_name), self.gen_loss_a2b))
        self.scales.append(tf.summary.scalar('{}/DLossB'.format(self.dataset_name), self.dis_loss_b))
        self.scalar_metric = tf.placeholder(tf.float32, None, name='scalar_metric')
        self.scales.append(tf.summary.scalar('{}/MetricA2B'.format(self.dataset_name), self.scalar_metric))
        self.scales = tf.summary.merge(self.scales)

        self.images.append(
            tf.summary.image('{}/{}_AReal({}x{})'.format(self.model_id, *self.output_size), self.real_a, max_outputs=1))
        self.images.append(
            tf.summary.image('{}/{}_BFake({}x{})'.format(self.model_id, *self.output_size), self.fake_b, max_outputs=1))
        self.images.append(
            tf.summary.image('{}/{}_BReal({}x{})'.format(self.model_id, *self.output_size), self.real_b, max_outputs=1))
        self.images = tf.summary.merge(self.images)

    def train(self, tf_config):
        """Train pix2pix"""
        alpha_transition = self.alpha_transition.assign(self.epoch_tensor / self.total_epoch)
        g_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.0, beta2=0.99).minimize(self.gen_loss_a2b,
                                                                                             var_list=self.gen_vars)
        d_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5, beta2=0.99).minimize(self.dis_loss_b,
                                                                                             var_list=self.dis_vars)
        # Initializing
        self.sess.run(tf.global_variables_initializer())
        # self.load(self.checkpoint_dir / self.test_model, self.train_saver)
        summary_writer = tf.summary.FileWriter(self.log_path, self.sess.graph)
        if self.process_size != 1 and self.process_size != 7:
            if self.is_transition:
                self.read_saver.restore(self.sess, str(self.read_path))
                self.out_saver.restore(self.sess, str(self.read_path))
            else:
                self.write_saver.restore(self.sess, str(self.read_path))
        train_generator = self.train_data_loader.get_data_generator(*self.output_size)
        train_size = self.train_data_loader.get_size()
        best_eval_metric = float('-inf')
        for epoch in range(self.pre_epoch, self.total_epoch):
            lr = self.scheduler_fn(epoch)
            alpha = epoch / self.total_epoch
            g_loss_sum = d_loss_sum = 0  # sum one epoch g_loss and d_loss
            best_g_loss = float('inf')  # request min g_loss
            best_real_a = best_fake_b = best_real_b = np.zeros(shape=(self.batch_size, *self.data_shape, 1))
            for step in range(train_size):
                _, _, batch_a, _, _, batch_b = next(train_generator)
                if self.is_transition and self.process_size != 0:
                    blurred_batch_b = zoom(batch_b, zoom=[1, 0.5, 0.5, 1], mode='nearest')
                    blurred_batch_b = zoom(blurred_batch_b, zoom=[1, 2, 2, 1], mode='nearest')
                    batch_b = alpha * batch_b + (1 - alpha) * blurred_batch_b
                # Update D network
                _, d_loss, _ = self.sess.run([d_optimizer, self.dis_loss_b, alpha_transition],
                                             feed_dict={self.real_a: batch_a, self.real_b: batch_b, self.lr_tensor: lr,
                                                        self.epoch_tensor: epoch})
                d_loss_sum += d_loss

                # Update G network and record fake outputs
                fake_b, _, g_loss, _ = self.sess.run([self.fake_b, g_optimizer, self.gen_loss_a2b, alpha_transition],
                                                     feed_dict={self.real_a: batch_a, self.real_b: batch_b,
                                                                self.lr_tensor: lr, self.epoch_tensor: epoch})
                if best_g_loss >= g_loss:  # min g_loss to show image
                    best_g_loss, best_real_a, best_fake_b, best_real_b = (g_loss, batch_a, fake_b, batch_b)
                g_loss_sum += g_loss
                print('{}/{} id:{}/{} Epoch:{:>3d}/{:<3d} Step:{:>4d}/{:<4d} g_loss:{:<5.5f} d_loss:{:<5.5f}'
                      .format(self.name, self.tag, self.model_id, self.max_id, epoch, self.total_epoch, step,
                              train_size, g_loss, d_loss))
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
                        metrics = {name: fn(np.array(nii_b), np.array(fake_nii_b)) for name, fn in
                                   self.metrics_fn.items()}
                        eval_metric_sum += float(metrics['ssim_metrics'])
                        num_eval_nii += 1
                        nii_b = list()
                        fake_nii_b = list()
                    current_path_b = path_b
                    if step == eval_size:  # finnish eval
                        break
                fake_b = self.sess.run(self.fake_b, feed_dict={self.real_a: batch_a})
                nii_b.append(batch_b[0, :, :, self.out_channels // 2])
                fake_nii_b.append(fake_b[0, :, :, self.out_channels // 2])

            # draw summary
            images = self.sess.run(self.images, feed_dict={self.real_a: best_real_a, self.fake_b: best_fake_b,
                                                           self.real_b: best_real_b})
            scales = self.sess.run(self.scales,
                                   feed_dict={self.lr_tensor: lr, self.gen_loss_a2b: g_loss_sum / train_size,
                                              self.dis_loss_b: d_loss_sum / train_size,
                                              self.scalar_metric: eval_metric_sum / num_eval_nii})
            summary_writer.add_summary(images, epoch)
            summary_writer.add_summary(scales, epoch)

            # save model
            if best_eval_metric <= eval_metric_sum:
                self.save(Path(self.save_path) / 'best', self.write_saver, epoch)
                best_eval_metric = eval_metric_sum
            # if epoch % self.save_freq == 0:
            #     self.save(self.checkpoint_dir / 'train', self.write_saver, epoch)
