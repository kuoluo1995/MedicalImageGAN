import numpy as np
import tensorflow as tf
from pathlib import Path
from models.base_gan_model import BaseGanModel
from models.loss_funcation import abs_loss
from utils import nii_utils
from utils.image_utils import ImagePool, save_images


class CycleGAN(BaseGanModel):
    def __init__(self, **kwargs):
        BaseGanModel.__init__(self, **kwargs)
        self.L1_lambda = self.kwargs['model']['L1_lambda']
        self.pool = ImagePool(self.kwargs['model']['maxsize'])
        self.real_a = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.in_channels],
                                     name='realA')
        self.real_b = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.out_channels],
                                     name='realB')
        self.fake_a = None
        self.fake_b = None
        self.fake_a_ = None
        self.fake_b_ = None
        self.G_loss_a2b = None
        self.G_loss_b2a = None
        self.G_loss = None

        self.fake_a_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size[0], self.image_size[1], self.in_channels],
                                            name='fakeA')
        self.fake_b_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size[0], self.image_size[1], self.out_channels],
                                            name='fakeB')
        self.D_loss_real_a = None
        self.D_loss_fake_a = None
        self.D_loss_a = None
        self.D_loss_real_b = None
        self.D_loss_fake_b = None
        self.D_loss_b = None
        self.D_loss = None
        self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
        self.G_optimizer = None
        self.D_optimizer = None
        self.G_sum = None
        self.D_sum = None

        self.test_a = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.in_channels],
                                     name='testA')
        self.test_b = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.out_channels],
                                     name='testB')
        self.test_G_a = None
        self.test_G_b = None

    def build_model(self, **kwargs):
        self.fake_b = self.generator_class(self.real_a, name='generatorA2B', **self.__dict__)
        self.fake_a_ = self.generator_class(self.fake_b, name='generatorB2A', **self.__dict__)
        self.fake_a = self.generator_class(self.real_b, reuse=True, name='generatorB2A', **self.__dict__)
        self.fake_b_ = self.generator_class(self.fake_a, reuse=True, name='generatorA2B', **self.__dict__)

        D_fake_b = self.discriminator_class(self.fake_b, name='discriminatorB', **self.__dict__)
        D_fake_a = self.discriminator_class(self.fake_a, name='discriminatorA', **self.__dict__)

        self.G_loss_a2b = self.loss_fn(D_fake_b, tf.ones_like(D_fake_b)) + \
                          self.L1_lambda * abs_loss(self.real_a, self.fake_a_) + \
                          self.L1_lambda * abs_loss(self.real_b, self.fake_b_)
        self.G_loss_b2a = self.loss_fn(D_fake_a, tf.ones_like(D_fake_a)) + \
                          self.L1_lambda * abs_loss(self.real_a, self.fake_a_) + \
                          self.L1_lambda * abs_loss(self.real_b, self.fake_b_)

        self.G_loss = self.loss_fn(D_fake_a, tf.ones_like(D_fake_a)) + \
                      self.loss_fn(D_fake_b, tf.ones_like(D_fake_b)) + \
                      self.L1_lambda * abs_loss(self.real_a, self.fake_a_) + \
                      self.L1_lambda * abs_loss(self.real_b, self.fake_b_)

        D_real_a = self.discriminator_class(self.real_a, reuse=True, name='discriminatorA', **self.__dict__)
        D_real_b = self.discriminator_class(self.real_b, reuse=True, name='discriminatorB', **self.__dict__)
        D_fake_a = self.discriminator_class(self.fake_a_sample, reuse=True, name='discriminatorA', **self.__dict__)
        D_fake_b = self.discriminator_class(self.fake_b_sample, reuse=True, name='discriminatorB', **self.__dict__)

        self.D_loss_real_a = self.loss_fn(D_real_a, tf.ones_like(D_real_a))
        self.D_loss_fake_a = self.loss_fn(D_fake_a, tf.zeros_like(D_fake_a))
        self.D_loss_a = (self.D_loss_real_a + self.D_loss_fake_a) / 2

        self.D_loss_real_b = self.loss_fn(D_real_b, tf.ones_like(D_real_b))
        self.D_loss_fake_b = self.loss_fn(D_fake_b, tf.zeros_like(D_fake_b))
        self.D_loss_b = (self.D_loss_real_b + self.D_loss_fake_b) / 2

        self.D_loss = self.D_loss_a + self.D_loss_b

        train_vars = tf.trainable_variables()
        # for var in train_vars:
        #     print(var.name)
        G_vars = [var for var in train_vars if 'generator' in var.name]
        D_vars = [var for var in train_vars if 'discriminator' in var.name]

        self.G_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.G_loss, var_list=G_vars)
        self.D_optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.D_loss, var_list=D_vars)

        self.test_G_a = self.generator_class(self.test_b, reuse=True, name='generatorB2A', **self.__dict__)
        self.test_G_b = self.generator_class(self.test_a, reuse=True, name='generatorA2B', **self.__dict__)
        self.summary()

    def summary(self):
        fake_a_uint8 = tf.cast((self.fake_a + 1.0) * 255 // 2, tf.uint8)
        fake_b_uint8 = tf.cast((self.fake_b + 1.0) * 255 // 2, tf.uint8)
        fake_a_sum = tf.summary.image('{}/fakeA'.format(self.name), fake_a_uint8, max_outputs=1)
        fake_b_sum = tf.summary.image('{}/fakeB'.format(self.name), fake_b_uint8, max_outputs=1)

        G_loss_a2b_sum = tf.summary.scalar('{}/GLossA2B'.format(self.name), self.G_loss_a2b)
        G_loss_b2a_sum = tf.summary.scalar('{}/GLossB2A'.format(self.name), self.G_loss_b2a)
        G_loss_sum = tf.summary.scalar('{}/GLoss'.format(self.name), self.G_loss)
        self.G_sum = tf.summary.merge([G_loss_a2b_sum, G_loss_b2a_sum, G_loss_sum, fake_a_sum, fake_b_sum])

        D_loss_real_a_sum = tf.summary.scalar('{}/DLossRealA'.format(self.name), self.D_loss_real_a)
        D_loss_real_b_sum = tf.summary.scalar('{}/DLossRealB'.format(self.name), self.D_loss_real_b)
        D_loss_fake_a_sum = tf.summary.scalar('{}/DLossFakeA'.format(self.name), self.D_loss_fake_a)
        D_loss_fake_b_sum = tf.summary.scalar('{}/DLossFakeB'.format(self.name), self.D_loss_fake_b)
        D_loss_a_sum = tf.summary.scalar('{}/DLossA'.format(self.name), self.D_loss_a)
        D_loss_b_sum = tf.summary.scalar('{}/DLossB'.format(self.name), self.D_loss_b)
        D_loss_sum = tf.summary.scalar('{}/DLoss'.format(self.name), self.D_loss)
        self.D_sum = tf.summary.merge([D_loss_real_a_sum, D_loss_real_b_sum, D_loss_fake_a_sum, D_loss_fake_b_sum,
                                       D_loss_a_sum, D_loss_b_sum, D_loss_sum])

    def train(self):
        writer = tf.summary.FileWriter('./_logs/{}'.format(self.name), self.sess.graph)
        train_generator = self.data_generator_fn(self.train_path, self.batch_size, self.image_size, self.in_channels,
                                                 self.is_training)
        for epoch in range(self.epoch):
            lr = self.lr if epoch < self.epoch_step else self.lr * (self.epoch - epoch) / (self.epoch - self.epoch_step)
            train_data = next(train_generator)
            fake_a, fake_b, _, summary_str, G_loss = self.sess.run(
                [self.fake_a, self.fake_b, self.G_optimizer, self.G_sum, self.G_loss],
                feed_dict={self.real_a: train_data[0], self.real_b: train_data[1], self.learning_rate: lr})
            writer.add_summary(summary_str, epoch)
            _, summary_str, D_loss = self.sess.run([self.D_optimizer, self.D_sum, self.D_loss],
                                                   feed_dict={self.real_a: train_data[0], self.real_b: train_data[1],
                                                              self.learning_rate: lr, self.fake_a_sample: fake_a,
                                                              self.fake_b_sample: fake_b})
            writer.add_summary(summary_str, epoch)
            print('Epoch:{:>5d} G_loss:{:<5.5f} D_loss:{:<5.5f}'.format(epoch, G_loss, D_loss))
        if epoch % self.save_freq == 0:
            self.save(self.checkpoint_dir, epoch)

    # def load(self, checkpoint_dir):
    #     checkpoint_dir = Path(checkpoint_dir)
    #     ckpt = tf.train.get_checkpoint_state(str(checkpoint_dir / 'cycle_gan.ckpt'))
    #     if ckpt and ckpt.model_checkpoint_path:
    #         ckpt_name = Path(ckpt.model_checkpoint_path).stem
    #         self.saver.restore(self.sess, str(checkpoint_dir / ckpt_name))
    # def test(self, option):
    #     """Test cyclegan"""
    #     init_op = tf.global_variables_initializer()
    #     self.sess.run(init_op)
    #     sample_files = list()  # todo
    #
    #     # write html for visual comparison
    #     index_path = Path(option.test_dir) / '_index.html'
    #     index = open(index_path, 'w')
    #     index.write('<html><body><table><tr>')
    #     index.write('<th>name</th><th>input</th><th>output</th></tr>')
    #
    #     out_var, in_var = (self.test_A, self.test_B)
    #
    #     for sample_file in sample_files:
    #         print('Processing images: {}'.format(sample_file))
    #         sample_image = []  # todo
    #         sample_image = np.array(sample_image).astype(np.float32)
    #         image_path = Path(option.test_dir) / Path(sample_file).stem
    #         fake_image = self.sess.run(out_var, feed_dict={in_var: sample_image})
    #         save_images(fake_image, [1, 1], image_path)
    #         index.write('<td>{}</td>'.format(Path(sample_file).stem))
    #         index.write('<td><img src="{}"></td>'.format(sample_file))
    #         index.write('<td><img src="{}"></td>'.format(image_path))
    #         index.write('</tr>')
    #     index.close()
