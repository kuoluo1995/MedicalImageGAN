import numpy as np
import tensorflow as tf
from pathlib import Path

from models import get_model_fn, BaseModel
from models.utils.layers import down_sampling
from models.utils.loss_funcation import l1_loss
from utils import yaml_utils
from utils.nii_utils import nii_header_reader, nii_writer


class Pix2PixHDGAN(BaseModel):
    def __init__(self, **kwargs):
        BaseModel.__init__(self, **kwargs)
        self.kwargs = Pix2PixHDGAN._get_extend_kwargs(self.kwargs)
        model = self.kwargs['model']
        self._lambda = self.kwargs['model']['lambda']
        self.num_scales = self.kwargs['model']['num_scales']
        self.global_generator = get_model_fn('generator', out_channels=self.out_channels,
                                             **model['global_generator'])
        self.enhancer_generator = get_model_fn('generator', out_channels=self.out_channels,
                                               **model['enhancer_generator'])
        self.discriminator = get_model_fn('discriminator', out_channels=self.out_channels, **model['discriminator'])
        self.build_model()
        self.summary()
        self.train_saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def build_model(self):
        # train generator
        data_shape = self.data_shape
        self.real_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.in_channels], 'real_a')
        self.real_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], self.out_channels], 'real_b')
        real_a_scales = [None] * 3
        real_b_scales = [None] * 3
        for i in range(self.num_scales):
            with tf.name_scope('scale_' + str(i)):
                if i == 0:
                    real_a_scales[i] = self.real_a
                    real_b_scales[i] = self.real_b
                else:
                    with tf.name_scope('real_a'):
                        real_a_scales[i] = down_sampling(real_a_scales, i)
                    with tf.name_scope('real_b'):
                        real_b_scales[i] = down_sampling(real_b_scales, i)
        # Create coarse image
        coarse_image, coarse_feature_map = self.global_generator(self.real_a[1], is_training=True,
                                                                 name='coarse_generator')
        fake_b_scales = [None] * 3
        fake_b_scales.append(self.global_generator([self.real_a, coarse_feature_map], training=True))
        for i in range(1, 3):
            fake_b_scales.append(down_sampling(fake_b_scales[0], i))

        generator_total_loss = 0
        discriminator_total_loss = 0

        # Discriminate each scale
        for i in range(3):
            with tf.name_scope('scale_' + str(i)):
                # Define discriminators
                real_logit_b = self.discriminator([real_a_scales[i], real_b_scales[i]], training=True,
                                                  name='discriminator_b_' + str(i))
                fake_logit_b = self.discriminator([real_a_scales[i], fake_b_scales[i]], training=True, reuse=True,
                                                  name='discriminator_b_' + str(i))
                g_loss_a2b = self.loss_fn(fake_logit_b, tf.ones_like(fake_logit_b)) + self._lambda * l1_loss(
                    fake_b_scales[i], real_b_scales[i])
                generator_total_loss += g_loss_a2b
                d_loss_real_b = self.loss_fn(real_logit_b, tf.ones_like(real_logit_b))
                d_loss_fake_b = self.loss_fn(fake_logit_b, tf.zeros_like(fake_logit_b))
                d_loss_b = d_loss_real_b + d_loss_fake_b
                discriminator_total_loss += d_loss_b

    def summary(self):
        data_shape = self.data_shape
        self.image_real_a = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], 1], name='image_real_a')
        real_a_summary = tf.summary.image('{}/AReal'.format(self.dataset_name), self.image_real_a, max_outputs=1)

        self.image_fake_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], 1], name='image_fake_b')
        fake_b_summary = tf.summary.image('{}/BFake'.format(self.dataset_name), self.image_fake_b, max_outputs=1)

        self.image_real_b = tf.placeholder(tf.float32, [None, data_shape[0], data_shape[1], 1], name='image_real_b')
        real_b_summary = tf.summary.image('{}/BReal'.format(self.dataset_name), self.image_real_b, max_outputs=1)

        self.image_summary = tf.summary.merge([real_a_summary, real_b_summary, fake_b_summary])

        lr_summary = tf.summary.scalar('{}/LearningRate'.format(self.dataset_name), self.lr_tensor)
        self.scalar_g_loss = tf.placeholder(tf.float32, None, name='scalar_g_loss')
        g_loss_summary = tf.summary.scalar('{}/GLossA2B'.format(self.dataset_name), self.scalar_g_loss)
        self.scalar_d_loss = tf.placeholder(tf.float32, None, name='scalar_d_loss')
        d_loss_summary = tf.summary.scalar('{}/DLossB'.format(self.dataset_name), self.scalar_d_loss)
        self.scalar_metric = tf.placeholder(tf.float32, None, name='scalar_metric')
        eval_metric_summary = tf.summary.scalar('{}/MetricA2B'.format(self.dataset_name), self.scalar_metric)
        self.scalar_summary = tf.summary.merge([lr_summary, g_loss_summary, d_loss_summary, eval_metric_summary])

    def train(self):
        """Train pix2pix"""
        coarse_g_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.g_loss_a2b,
                                                                                        var_list=self.cg_vars)
        enhancer_g_optimiaze = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.g_loss_a2b,
                                                                                          var_list=self.eg_vars)
        d_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.d_loss_b, var_list=self.d_vars)
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
            g_loss_sum = d_loss_sum = 0  # sum one epoch g_loss and d_loss
            best_g_loss = float('inf')  # request min g_loss
            best_real_a = best_fake_b = best_real_b = np.zeros(
                shape=(self.batch_size, self.data_shape[0], self.data_shape[1], 1))
            for step in range(train_size):
                _, _, batch_a, _, _, batch_b = next(train_generator)
                # Update G network and record fake outputs
                fake_b, _, g_loss = self.sess.run([self._fake_b, g_optimizer, self.g_loss_a2b],
                                                  feed_dict={self.real_a: batch_a, self.real_b: batch_b,
                                                             self.lr_tensor: lr})
                if best_g_loss >= g_loss:  # min g_loss to show image
                    best_g_loss, best_real_a, best_fake_b, best_real_b = (g_loss, batch_a, fake_b, batch_b)
                g_loss_sum += g_loss
                # Update D network
                _, d_loss = self.sess.run([d_optimizer, self.d_loss_b],
                                          feed_dict={self.real_a: batch_a, self.real_b: batch_b, self.fake_b: fake_b,
                                                     self.lr_tensor: lr})
                d_loss_sum += d_loss
                print('{}/{} Epoch:{:>3d}/{:<3d} Step:{:>4d}/{:<4d} g_loss:{:<5.5f} d_loss:{:<5.5f}'
                      .format(self.name, self.tag, epoch, self.total_epoch, step, train_size, g_loss, d_loss))
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
                fake_b = self.sess.run(self.test_fake_b, feed_dict={self.test_a: batch_a})
                nii_b.append(batch_b[0, :, :, self.out_channels // 2])
                fake_nii_b.append(fake_b[0, :, :, self.out_channels // 2])

            # draw summary
            image_summary = self.sess.run(self.image_summary,
                                          feed_dict={self.image_real_a: best_real_a, self.image_fake_b: best_fake_b,
                                                     self.image_real_b: best_real_b})
            scalar_summary = self.sess.run(self.scalar_summary,
                                           feed_dict={self.lr_tensor: lr, self.scalar_g_loss: g_loss_sum / train_size,
                                                      self.scalar_d_loss: d_loss_sum / train_size,
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
            fake_b = self.sess.run(self.test_fake_b, feed_dict={self.test_a: batch_a})
            nii_b.append(batch_b[0, :, :, self.out_channels // 2])
            fake_nii_b.append(fake_b[0, :, :, self.out_channels // 2])
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