import numpy as np
from pathlib import Path

from configs.option import Option
from models.networks import *
from models.networks.patch_gan import patch_gan
from models.networks.unet import unet
from utils import yaml_utils
from utils.image_utils import load_data


class CycleGAN:
    def __init__(self, sess, option):
        option = Option(option)
        self.sess = sess
        self.options = option
        self.print_freq = option.print_freq
        self.save_freq = option.save_freq

        self.batch_size = option.batch_size
        self.image_size = option.dataset.image_size
        self.in_channels = option.model.in_channels
        self.out_channels = option.model.out_channels
        self.L1_lambda = option.model.l1_lambda
        self.is_training = option.phase == 'train'
        self.train_dir = yaml_utils.read(option.dataset.train_path)
        self.test_dir = yaml_utils.read(option.dataset.test_path)

        self.generator = unet
        self.discriminator = patch_gan
        self.criterionGAN = lsgan_loss
        net_options = {'batch_size': self.batch_size, 'image_size': self.image_size, 'out_channels': self.out_channels,
                       'G_channels': option.model.generator.channels, 'D_channels': option.model.discriminator.channels,
                       'is_training': self.is_training}
        self._build_model(net_options)
        self.saver = tf.train.Saver()

    def _build_model(self, net_options):
        self.real_data = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1],
                                                     self.in_channels + self.out_channels], name='real_A_and_B_images')
        real_A = self.real_data[:, :, :, :self.in_channels]
        real_A_sum = tf.summary.image('{}/real_A'.format(self.options.tag), real_A, max_outputs=1)
        self.fake_B = self.generator(real_A, name='generatorA2B', **net_options)
        fake_B_sum = tf.summary.image('{}/real_A2fake_B'.format(self.options.tag), self.fake_B, max_outputs=1)
        self.fake_A_ = self.generator(self.fake_B, name='generatorB2A', **net_options)
        DB_fake = self.discriminator(self.fake_B, name='discriminatorB', **net_options)

        real_B = self.real_data[:, :, :, self.in_channels:self.in_channels + self.out_channels]
        real_B_sum = tf.summary.image('{}/real_B'.format(self.options.tag), real_B, max_outputs=1)
        self.fake_A = self.generator(real_B, reuse=True, name='generatorB2A', **net_options)
        fake_A_sum = tf.summary.image('{}/real_B2fake_A'.format(self.options.tag), self.fake_A, max_outputs=1)
        self.fake_B_ = self.generator(self.fake_A, reuse=True, name='generatorA2B', **net_options)
        DA_fake = self.discriminator(self.fake_A, name='discriminatorA', **net_options)  # todo read cycle

        GA2B_loss = self.criterionGAN(DB_fake, tf.ones_like(DB_fake)) + \
                    self.L1_lambda * abs_criterion(real_A, self.fake_A_) + \
                    self.L1_lambda * abs_criterion(real_B, self.fake_B_)
        GA2B_loss_sum = tf.summary.scalar('{}/GA2B_loss'.format(self.options.tag), GA2B_loss)

        GB2A_loss = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) + \
                    self.L1_lambda * abs_criterion(real_A, self.fake_A_) + \
                    self.L1_lambda * abs_criterion(real_B, self.fake_B_)
        GB2A_loss_sum = tf.summary.scalar('{}/GB2A_loss'.format(self.options.tag), GB2A_loss)

        self.G_loss = self.criterionGAN(DA_fake, tf.ones_like(DA_fake)) + \
                      self.criterionGAN(DB_fake, tf.ones_like(DB_fake)) + \
                      self.L1_lambda * abs_criterion(real_A, self.fake_A_) + \
                      self.L1_lambda * abs_criterion(real_B, self.fake_B_)
        G_loss_sum = tf.summary.scalar('{}/G_loss'.format(self.options.tag), self.G_loss)
        self.G_sum = tf.summary.merge(
            [GA2B_loss_sum, GB2A_loss_sum, G_loss_sum, real_A_sum, fake_A_sum, real_B_sum, fake_B_sum])

        DA_real = self.discriminator(real_A, reuse=True, name='discriminatorA', **net_options)
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size[0], self.image_size[1], self.in_channels],
                                            name='fake_A_sample')
        DA_fake_sample = self.discriminator(self.fake_A_sample, reuse=True, name='discriminatorA', **net_options)
        DA_loss_real = self.criterionGAN(DA_real, tf.ones_like(DA_real))
        DA_loss_real_sum = tf.summary.scalar('{}/DA_loss_real'.format(self.options.tag), DA_loss_real)

        DA_loss_fake = self.criterionGAN(DA_fake_sample, tf.zeros_like(DA_fake_sample))
        DA_loss_fake_sum = tf.summary.scalar('{}/DA_loss_fake'.format(self.options.tag), DA_loss_fake)

        DA_loss = (DA_loss_real + DA_loss_fake) / 2
        DA_loss_sum = tf.summary.scalar('{}/DA_loss'.format(self.options.tag), DA_loss)

        DB_real = self.discriminator(real_B, reuse=True, name='discriminatorB', **net_options)
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size[0], self.image_size[1], self.out_channels],
                                            name='fake_B_sample')
        DB_fake_sample = self.discriminator(self.fake_B_sample, reuse=True, name='discriminatorB', **net_options)
        DB_loss_real = self.criterionGAN(DB_real, tf.ones_like(DB_real))
        DB_loss_real_sum = tf.summary.scalar('{}/DB_loss_real'.format(self.options.tag), DB_loss_real)

        DB_loss_fake = self.criterionGAN(DB_fake_sample, tf.zeros_like(DB_fake_sample))
        DB_loss_fake_sum = tf.summary.scalar('{}/DB_loss_fake'.format(self.options.tag), DB_loss_fake)

        DB_loss = (DB_loss_real + DB_loss_fake) / 2
        DB_loss_sum = tf.summary.scalar('{}/DB_loss'.format(self.options.tag), DB_loss)

        self.D_loss = DA_loss + DB_loss
        D_loss_sum = tf.summary.scalar('{}/D_loss'.format(self.options.tag), self.D_loss)

        self.D_sum = tf.summary.merge(
            [DA_loss_sum, DA_loss_real_sum, DA_loss_fake_sum, DB_loss_sum, DB_loss_real_sum, DB_loss_fake_sum,
             D_loss_sum])

        self.test_A = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.in_channels],
                                     name='test_A')
        self.test_GB = self.generator(self.test_A, reuse=True, name='generatorA2B', **net_options)
        self.test_B = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], self.out_channels],
                                     name='test_B')
        self.test_GA = self.generator(self.test_B, reuse=True, name='generatorB2A', **net_options)

        train_vars = tf.trainable_variables()
        G_vars = [var for var in train_vars if 'generator' in var.name]
        D_vars = [var for var in train_vars if 'discriminator' in var.name]

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.G_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.G_loss, var_list=G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.lr, beta1=0.5).minimize(self.D_loss, var_list=D_vars)

    def train(self):
        """Train cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        writer = tf.summary.FileWriter('../LiverDiscriminator/logs/{}'.format(self.options.tag), self.sess.graph)

        counter = 1
        for epoch in range(self.options.epoch):
            # batch_idxs = len(data_A) // self.batch_size
            lr = self.options.lr if epoch < self.options.epoch_step else self.options.lr * (
                    self.options.epoch - epoch) / (self.options.epoch - self.options.epoch_step)
            train_dirs = [[self.train_dir['A'][i], self.train_dir['B'][i]] for i in range(len(self.train_dir['A']))]
            np.random.shuffle(train_dirs)
            for i, train_data in enumerate(train_dirs):
                nii_images = load_data(train_data, self.image_size, self.batch_size, self.is_training)
                for j, slice_ in enumerate(nii_images):
                    batch_images = np.array([slice_],dtype=np.float32)
                    # Update G network and record fake outputs
                    fake_A, fake_B, _, summary_str, G_loss = self.sess.run(
                        [self.fake_A, self.fake_B, self.G_optim, self.G_sum, self.G_loss],
                        feed_dict={self.real_data: batch_images, self.lr: lr})
                    writer.add_summary(summary_str, counter)
                    # [fake_A, fake_B] = self.pool([fake_A, fake_B])
                    # Update D network
                    _, summary_str, D_loss = self.sess.run([self.D_optim, self.D_sum, self.D_loss],
                                                           feed_dict={self.real_data: batch_images, self.lr: lr,
                                                                      self.fake_A_sample: fake_A,
                                                                      self.fake_B_sample: fake_B})
                    writer.add_summary(summary_str, counter)
                    counter += 1
                    print('Epoch:{:>2d}/{:<3d};Step:{:>3d}/{:<3d};Counter:{:>3d}/{:<3d} G_loss:{:<5.5f} D_loss:{:<5.5f}'
                          .format(epoch, self.options.epoch, i, len(train_dirs), j, len(nii_images), G_loss, D_loss))
                # if (epoch * len(train_dirs) + i) % self.print_freq == 0:
                #     self.sample_model(self.options.model.sample_dir, epoch, i)
                if (epoch * len(train_dirs) + i) % self.save_freq == 0:
                    self.save(self.options.model.checkpoint_dir, counter)

    def save(self, checkpoint_dir, counter):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.saver.save(self.sess, str(checkpoint_dir / 'cycle_gan.ckpt'), global_step=counter)

    # def sample_model(self, sample_dir, epoch, step):
    #     test_dirs = [[self.test_dir['A'][i], self.test_dir['B'][i]] for i in range(len(self.test_dir['A']))]
    #     np.random.shuffle(test_dirs)
    #     for i, test_data in enumerate(test_dirs):
    #         nii_images = load_data(test_data, self.image_size, self.is_training)
    #         test_nii_a = list()
    #         test_nii_b = list()
    #         for slice_ in nii_images:
    #             fake_A, fake_B = self.sess.run([self.fake_A, self.fake_B], feed_dict={self.real_data: slice_})
    #             test_nii_a.append(np.squeeze(fake_A))
    #             test_nii_b.append(np.squeeze(fake_B))
    #
    #         test_nii_a = np.array(test_nii_a).transpose((2, 0, 1))
    #         header_a = nii_utils.nii_header_reader(test_data[0])
    #         nii_utils.nii_writer('{}/A/{:^2d}_{:^4d}/{}.nii'.format(sample_dir, epoch, step, i), header_a, test_nii_a)
    #
    #         test_nii_b = np.array(test_nii_b).transpose((2, 0, 1))
    #         header_b = nii_utils.nii_header_reader(test_data[1])
    #         nii_utils.nii_writer('{}/B/{:^2d}_{:^4d}/{}.nii'.format(sample_dir, epoch, step, i), header_b, test_nii_b)
    #
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
