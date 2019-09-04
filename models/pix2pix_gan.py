import tensorflow as tf
from models.base_gan_model import BaseGanModel
from models.utils.loss_funcation import l1_loss
from data_loader import get_epoch_step


class Pix2PixGAN(BaseGanModel):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self._lambda = self.kwargs['model']['lambda']
        self.build_model()
        self.summary()
        self.saver = tf.train.Saver()

    def build_model(self):
        # train generator
        self.realA = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.in_channels],
                                    name='realA')
        self.realB = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.out_channels],
                                    name='realB')
        self.fakeB = self.generator(self.realA, name='generatorA2B')
        fakeB_logit = self.discriminator(self.fakeB, name='discriminatorB')
        self.g_lossA2B = self.loss_fn(fakeB_logit, tf.ones_like(fakeB_logit)) + self._lambda * l1_loss(self.realB,
                                                                                                       self.fakeB)

        # train discriminator
        self.fakeB_sample = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.in_channels],
                                           name='fakeB')
        realB_logit = self.discriminator(self.realB, reuse=True, name='discriminatorB')
        fakeB_logit = self.discriminator(self.fakeB_sample, reuse=True, name='discriminatorB')

        self.d_loss_realB = self.loss_fn(realB_logit, tf.ones_like(realB_logit))
        self.d_loss_fakeB = self.loss_fn(fakeB_logit, tf.zeros_like(fakeB_logit))
        self.d_lossB = self.d_loss_realB + self.d_loss_fakeB

        train_vars = tf.trainable_variables()
        self.g_vars = [var for var in train_vars if 'generator' in var.name]
        self.d_vars = [var for var in train_vars if 'discriminator' in var.name]

    def summary(self):
        realA_sum = tf.summary.image('{}/{}/{}/AReal'.format(self.dataset_name, self.name, self.tag), self.realA,
                                     max_outputs=1)
        fakeB_sum = tf.summary.image('{}/{}/{}/BFake'.format(self.dataset_name, self.name, self.tag), self.fakeB,
                                     max_outputs=1)
        realB_sum = tf.summary.image('{}/{}/{}/BReal'.format(self.dataset_name, self.name, self.tag), self.realB,
                                     max_outputs=1)

        g_loss_A2B_sum = tf.summary.scalar('{}/{}/{}/GLossA2B'.format(self.dataset_name, self.name, self.tag),
                                           self.g_lossA2B)
        self.g_sum = tf.summary.merge([g_loss_A2B_sum, realA_sum, realB_sum, fakeB_sum])

        d_loss_realB_sum = tf.summary.scalar('{}/{}/{}/DLossRealB'.format(self.dataset_name, self.name, self.tag),
                                             self.d_loss_realB)
        d_loss_fakeB_sum = tf.summary.scalar('{}/{}/{}/DLossFakeB'.format(self.dataset_name, self.name, self.tag),
                                             self.d_loss_fakeB)
        d_loss_B_sum = tf.summary.scalar('{}/{}/{}/DLossB'.format(self.dataset_name, self.name, self.tag), self.d_lossB)

        lr_sum = tf.summary.scalar('{}/{}/{}/LearningRate'.format(self.dataset_name, self.name, self.tag),
                                   self.lr_tensor)
        self.d_sum = tf.summary.merge([d_loss_realB_sum, d_loss_fakeB_sum, d_loss_B_sum, lr_sum])

    def train(self):
        """Train cyclegan"""
        g_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.g_lossA2B, var_list=self.g_vars)
        d_optimizer = tf.train.AdamOptimizer(self.lr_tensor, beta1=0.5).minimize(self.d_lossB, var_list=self.d_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        writer = tf.summary.FileWriter('../tensorboard_logs/{}/{}/{}'.format(self.dataset_name, self.name, self.tag),
                                       self.sess.graph)

        train_generator = self.data_loader(self.train_dataset, self.batch_size, self.image_size, self.in_channels,
                                           self.is_training)
        epoch_step = get_epoch_step(self.train_dataset)
        for epoch in range(self.epoch):
            lr = self.scheduler_fn(epoch)
            for step in range(epoch_step):
                realA, realB = next(train_generator)

                # Update G network and record fake outputs
                fakeB, _, g_sum, g_loss = self.sess.run([self.fakeB, g_optimizer, self.g_sum, self.g_lossA2B],
                                                        feed_dict={self.realA: realA, self.realB: realB,
                                                                   self.lr_tensor: lr})
                writer.add_summary(g_sum, epoch * epoch_step + step)

                # Update D network
                _, d_sum, d_loss = self.sess.run([d_optimizer, self.d_sum, self.d_lossB],
                                                 feed_dict={self.realB: realB, self.fakeB_sample: fakeB,
                                                            self.lr_tensor: lr})
                writer.add_summary(d_sum, epoch * epoch_step + step)
                print('Epoch:{:>3d}/{:<3d} Step:{:>4d}/{:<4d} g_loss:{:<5.5f} d_loss:{:<5.5f}'.format(epoch, self.epoch,
                                                                                                      step, epoch_step,
                                                                                                      g_loss, d_loss))
            if epoch % self.save_freq == 0:
                self.save(self.checkpoint_dir, epoch)

    # def test(self):
    #     test_generator = self.data_loader(self.test_dataset, self.batch_size, self.image_size, self.in_channels,
    #                                       self.is_training)
    #     self.load(self.checkpoint_dir)
    #     epoch_step = get_epoch_step(self.test_dataset)
    #     for epoch in range(self.test_size):
    #         test_data = next(test_generator)
