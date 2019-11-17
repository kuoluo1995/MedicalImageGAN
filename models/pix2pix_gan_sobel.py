import numpy as np
import tensorflow as tf

from models.base_gan_model import BaseGanModel
from models.pix2pix_gan import Pix2PixGAN


class Pix2PixGANSobel(Pix2PixGAN):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self._lambda = self.kwargs['model']['lambda']
        self.build_model()
        self.summary()
        self.train_saver = tf.train.Saver()
        self.best_saver = tf.train.Saver()

    def train(self):
        """Train pix2pix"""
        g_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5, beta2=0.999).minimize(self.g_loss_a2b,
                                                                                              var_list=self.g_vars)
        d_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5, beta2=0.999).minimize(self.d_loss_b,
                                                                                              var_list=self.d_vars)
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
            image_summary = self.sess.run(self.image_summary, feed_dict={self.image_real_a: best_real_a[:, :, :, 0:1],
                                                                         self.image_fake_b: best_fake_b,
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
