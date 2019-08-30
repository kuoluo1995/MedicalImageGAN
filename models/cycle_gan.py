import tensorflow as tf
from models.base_gan_model import BaseGanModel
from models.utils.loss_funcation import l1_loss
from data_loader import get_epoch_step
from utils.image_utils import ImagePool


class CycleGAN(BaseGanModel):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self._lambda = self.kwargs['model']['lambda']
        self.image_pool = ImagePool(self.kwargs['model']['maxsize'])
        self.build_model()
        self.summary()
        self.saver = tf.train.Saver()

    def build_model(self):
        # train generator
        self.realA = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.in_channels],
                                    name='realA')
        self.fakeB = self.generator(self.realA, name='generatorA2B')
        self.recA = self.generator(self.fakeB, name='generatorB2A')

        self.realB = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.out_channels],
                                    name='realB')
        self.fakeA = self.generator(self.realB, reuse=True, name='generatorB2A')
        self.recB = self.generator(self.fakeA, reuse=True, name='generatorA2B')

        fakeA_logit = self.discriminator(self.fakeA, name='discriminatorA')
        fakeB_logit = self.discriminator(self.fakeB, name='discriminatorB')
        with tf.variable_scope('cycle_loss'):
            cycle_loss = self._lambda * l1_loss(self.realA, self.recA) + self._lambda * l1_loss(self.realB, self.recB)
        self.g_loss_A2B = self.loss_fn(fakeB_logit, tf.ones_like(fakeB_logit)) + cycle_loss
        self.g_loss_B2A = self.loss_fn(fakeA_logit, tf.ones_like(fakeA_logit)) + cycle_loss

        self.g_loss = self.loss_fn(fakeA_logit, tf.ones_like(fakeA_logit)) + \
                      self.loss_fn(fakeB_logit, tf.ones_like(fakeB_logit)) + cycle_loss
        # train discriminator
        self.fakeA_sample = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.in_channels],
                                           name='fakeA')
        self.fakeB_sample = tf.placeholder(tf.float32,
                                           [None, self.image_size[0], self.image_size[1], self.out_channels],
                                           name='fakeB')
        realA_logit = self.discriminator(self.realA, reuse=True, name='discriminatorA')
        fakeA_logit = self.discriminator(self.fakeA_sample, reuse=True, name='discriminatorA')

        realB_logit = self.discriminator(self.realB, reuse=True, name='discriminatorB')
        fakeB_logit = self.discriminator(self.fakeB_sample, reuse=True, name='discriminatorB')

        self.d_loss_realA = self.loss_fn(realA_logit, tf.ones_like(realA_logit))
        self.d_loss_fakeA = self.loss_fn(fakeA_logit, tf.zeros_like(fakeA_logit))
        self.d_loss_A = self.d_loss_realA + self.d_loss_fakeA

        self.d_loss_realB = self.loss_fn(realB_logit, tf.ones_like(realB_logit))
        self.d_loss_fakeB = self.loss_fn(fakeB_logit, tf.zeros_like(fakeB_logit))
        self.d_loss_B = self.d_loss_realB + self.d_loss_fakeB

        self.d_loss = self.d_loss_A + self.d_loss_B

        train_vars = tf.trainable_variables()
        self.g_vars = [var for var in train_vars if 'generator' in var.name]
        self.d_vars = [var for var in train_vars if 'discriminator' in var.name]

    def summary(self):
        realA_sum = tf.summary.image('{}/realA'.format(self.tag), self.realA, max_outputs=1)
        realB_sum = tf.summary.image('{}/realB'.format(self.tag), self.realB, max_outputs=1)
        fakeA_sum = tf.summary.image('{}/fakeA'.format(self.tag), self.fakeA, max_outputs=1)
        fakeB_sum = tf.summary.image('{}/fakeB'.format(self.tag), self.fakeB, max_outputs=1)
        g_loss_A2B_sum = tf.summary.scalar('{}/GLossA2B'.format(self.tag), self.g_loss_A2B)
        g_loss_B2A_sum = tf.summary.scalar('{}/GLossB2A'.format(self.tag), self.g_loss_B2A)
        g_loss_sum = tf.summary.scalar('{}/GLoss'.format(self.tag), self.g_loss)
        self.g_sum = tf.summary.merge(
            [g_loss_A2B_sum, g_loss_B2A_sum, g_loss_sum, realA_sum, fakeA_sum, realB_sum, fakeB_sum])

        d_loss_realA_sum = tf.summary.scalar('{}/DLossRealA'.format(self.tag), self.d_loss_realA)
        d_loss_realB_sum = tf.summary.scalar('{}/DLossRealB'.format(self.tag), self.d_loss_realB)
        d_loss_fakeA_sum = tf.summary.scalar('{}/DLossFakeA'.format(self.tag), self.d_loss_fakeA)
        d_loss_fakeB_sum = tf.summary.scalar('{}/DLossFakeB'.format(self.tag), self.d_loss_fakeB)
        d_loss_A_sum = tf.summary.scalar('{}/DLossA'.format(self.tag), self.d_loss_A)
        d_loss_B_sum = tf.summary.scalar('{}/DLossB'.format(self.tag), self.d_loss_B)
        d_loss_sum = tf.summary.scalar('{}/DLoss'.format(self.tag), self.d_loss)
        self.d_sum = tf.summary.merge([d_loss_realA_sum, d_loss_realB_sum, d_loss_fakeA_sum, d_loss_fakeB_sum,
                                       d_loss_A_sum, d_loss_B_sum, d_loss_sum])

    def train(self):
        """Train cyclegan"""
        lr_tensor = tf.placeholder(tf.float32, None, name='learning_rate')
        g_optimizer = tf.train.AdamOptimizer(lr_tensor, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        d_optimizer = tf.train.AdamOptimizer(lr_tensor, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        writer = tf.summary.FileWriter('../LiverDiscriminator/logs/{}'.format(self.tag), self.sess.graph)

        train_generator = self.data_loader(self.train_dataset, self.batch_size, self.image_size, self.in_channels,
                                           self.is_training)
        epoch_step = get_epoch_step(self.train_dataset)
        for epoch in range(self.epoch):
            lr = self.scheduler_fn(epoch)
            for step in range(epoch_step):
                realA, realB = next(train_generator)

                # Update G network and record fake outputs
                fakeA, fakeB, _, g_sum, g_loss = self.sess.run(
                    [self.fakeA, self.fakeB, g_optimizer, self.g_sum, self.g_loss],
                    feed_dict={self.realA: realA, self.realB: realB, lr_tensor: lr})
                writer.add_summary(g_sum, epoch * epoch_step + step)

                fakeA, fakeB = self.image_pool(fakeA, fakeB)

                # Update D network
                _, d_sum, d_loss = self.sess.run([d_optimizer, self.d_sum, self.d_loss],
                                                 feed_dict={self.realA: realA, self.realB: realB,
                                                            self.fakeA_sample: fakeA, self.fakeB_sample: fakeB,
                                                            lr_tensor: lr})
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
