# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
from six.moves import xrange
import numpy as np

from ops import batch_norm, conv2d, conv2d_transpose, lrelu, linear
from utils import get_image, save_images, get_neighbours

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))


class DCGAN(object):
    def __init__(self,
                 sess,
                 image_size,
                 batch_size=64,
                 sample_size=64,
                 lowres=8,
                 z_dim=100,
                 gf_dim=64,
                 df_dim=64,
                 gfc_dim=1024,
                 dfc_dim=1024,
                 c_dim=3,
                 checkpoint_dir=None,
                 lamda=0.01,
                 center_scale=0.25):

        # Currently, image size must be a (power of 2) and (8 or higher).
        assert(image_size & (image_size - 1) == 0 and image_size >= 8)
        print("========INITIALIZING THE WENGER DCGAN============")
        # INIT from train-dcgan!
        self.sess = sess
        self.is_crop = False
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]
        self.lowres = lowres
        self.lowres_size = image_size // lowres
        self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        self.lamda = lamda
        self.c_dim = c_dim
        self.center_scale = center_scale

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bns = [
            batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]

        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [
            batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]

        self.checkpoint_dir = checkpoint_dir

        print("========CALLING SETUP TO BUILD MODEL!============")
        self.setup()

        self.model_name = "DCGAN.model"

    def setup(self):
        self.training_bool = tf.placeholder(tf.bool, name='training_bool')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')

        # 2x4 calculates mean of pixels!
        self.lowres_images = tf.reduce_mean(tf.reshape(self.images,
            [self.batch_size, self.lowres_size, self.lowres,
             self.lowres_size, self.lowres, self.c_dim]), [2, 4])

        # initial distribution that G(z) uses
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        #intialize instance of generator + lowres generator for contextual loss calculation
        self.G = self.generator(self.z)
        self.lowres_G = tf.reduce_mean(tf.reshape(self.G,
            [self.batch_size, self.lowres_size, self.lowres,
             self.lowres_size, self.lowres, self.c_dim]), [2, 4])

        #need two discriminators, one for batches of images from our training data, one for
        #batches of images outputted by gen
        self.D, self.D_logits = self.discriminator(self.images)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        #discriminator loss for training data: cross entropy btwn discriminator pred and all ones (bc real)
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))

        #discriminator loss for generator outputs: cross entropy btwn discriminator pred and all zeros (bc fake)
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_)))

        # generator loss wants D to be wrong! and D says all real!
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_)))

        #generator loss: cross entropy between discriminator and all ones (want to fool discriminator
        #into thinking its outputs are real)
        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        #sum discriminator losses
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        #completion variable shtuff

        #mask to be completed (1s and 0s in shape of image)
        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        self.lowres_mask = tf.placeholder(tf.float32, self.lowres_shape, name='lowres_mask')



        self.weighted_contextual_loss = np.full(self.image_shape, 0.5, dtype=np.float32)
        for row_index in range(self.image_shape[0]):
            for col_index in range(self.image_shape[1]):
                # make masked
                if self.mask[row_index][col_index] == [0, 0, 0]:
                    self.weighted_contextual_loss[row_index][col_index] = [0.0, 0.0, 0.0]
                else: # not masked, if near mask then > 0.5 weight, add 0.5 for every closeby
                    weight = get_neighbours(row_index, col_index, self.mask)
                    self.weighted_contextual_loss[row_index][col_index] += [weight, weight, weight]

        #define contextual loss as pixel difference between mask * generator output and mask * image to infill
        # added weighted_contextual_loss
        # self.contextual_loss = tf.reduce_sum(
        #     tf.contrib.layers.flatten(
        #         tf.abs(tf.multiply(self.weighted_contextual_loss, tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1))


        # mi = tf.multiply(self.mask, self.images)
        # mG = tf.multiply(self.mask, self.G)
        # diff = mG - mi
        # wdiff = tf.multiply(self.weighted_contextual_loss, diff)
        # self.contextual_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(wdiff)), 1)

        mi = tf.multiply(self.mask, self.images)
        mG = tf.multiply(self.mask, self.G)
        # mi = tf.multiply(self.weighted_contextual_loss, self.images)
        # mG = tf.multiply(self.weighted_contextual_loss, self.G)
        diff = mG - mi
        self.contextual_loss = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs(diff)), 1)


        #as suggested by GAN implementations, add on same pixel difference for low res versions to include "bigger picture"
        self.contextual_loss += tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.lowres_mask, self.lowres_G) - tf.multiply(self.lowres_mask, self.lowres_images))), 1)

        # improve loss function to help smooth the mask boundary
        # TODO only works for center mask

        l = int(self.image_size*self.center_scale)
        u = int(self.image_size*(1.0-self.center_scale))

        # take G(z) the pixels in the outer part inside the mask
        generated_inner_borderline = np.zeros(self.image_shape).astype(np.float32)

        generated_inner_borderline[l, l:u, :] = 1.0
        generated_inner_borderline[l:u, u-1, :] = 1.0
        generated_inner_borderline[u-1, l:u, :] = 1.0
        generated_inner_borderline[l:u, l, :] = 1.0

        # take a second cut out inside the mask
        second_inner_borderline = np.zeros(self.image_shape).astype(np.float32)
        second_inner_borderline[l+1, l:u, :] = 1.0
        second_inner_borderline[l:u, u-2, :] = 1.0
        second_inner_borderline[u-2, l:u, :] = 1.0
        second_inner_borderline[l:u, l+1, :] = 1.0

        masked_image_outerborder = np.zeros(self.image_shape).astype(np.float32)
        masked_image_outerborder[l-1, l:u, :] = 1.0
        masked_image_outerborder[l:u, u, :] = 1.0
        masked_image_outerborder[u, l:u, :] = 1.0
        masked_image_outerborder[l:u, l-1, :] = 1.0

        # print(tf.abs(tf.multiply(generated_inner_borderline, self.G)))
        # print(tf.abs(tf.multiply(masked_image_outerborder, self.images)))

        self.blending_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(generated_inner_borderline, self.G) - tf.multiply(masked_image_outerborder, self.images))), 1)

        self.blending_loss += tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(second_inner_borderline, self.G) - tf.multiply(masked_image_outerborder, self.images))), 1)

        #to make sure we don't pick a G(z) that just doesn't look realistic, include perceptual loss (same loss as generator)
        #can be thought of as ensuring this G(z) fools the discriminator
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lamda*self.perceptual_loss + self.blending_loss
        #we will minimize loss function L = c + wz using gradient descent
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)

    def train(self, config):
        data = dataset_files(config.dataset)
        assert(len(data) > 0)
        np.random.shuffle(data)


        #optimizing parameters
        optimizing_discriminator = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        optimizing_generator = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        # get tf vars
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()


        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.file_writer = tf.summary.FileWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        sample_files = data[0:self.sample_size]

        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("========using existing model!!!============")
        else:
            print("========no checkpoint, initing new model!!!===========")

        #go through each epoch, sample some images and run optimizers to update network
        for epoch in xrange(config.epoch):
            data = dataset_files(config.dataset)
            batch_ids = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_ids):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([optimizing_discriminator, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z, self.training_bool: True })
                self.file_writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([optimizing_generator, self.g_sum],
                    feed_dict={ self.z: batch_z, self.training_bool: True })
                self.file_writer.add_summary(summary_str, counter)

                # Run optimizing_generator twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([optimizing_generator, self.g_sum],
                    feed_dict={ self.z: batch_z, self.training_bool: True })
                self.file_writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.training_bool: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.training_bool: False})
                errG = self.g_loss.eval({self.z: batch_z, self.training_bool: False})

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] seconds running: {:4.4f}, Discriminator_loss: {:.8f}, Generator_loss: {:.8f}".format(
                    epoch, idx, batch_ids, time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 1000) == 1:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.G, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.training_bool: False}
                    )
                    save_images(samples, [8, 8],
                                './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)


    def complete(self, config):
        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)
        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('logs')

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)

        number_of_images = len(config.imgs)
        lowres_mask = np.zeros(self.lowres_shape)

        batch_ids = int(np.ceil(number_of_images/self.batch_size))

        if config.maskType == 'random':
            fraction_masked = 0.8
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':
            assert(config.centerScale <= 0.5)

            # change centerScale to make diff size
            mask = np.ones(self.image_shape)
            lower_mask = int(self.image_size*config.centerScale)
            upper_mask = int(self.image_size*(1.0-config.centerScale))
            mask[lower_mask:upper_mask, lower_mask:upper_mask, :] = 0.0
        # elif config.maskType == 'left':
        #     mask = np.ones(self.image_shape)
        #     c = self.image_size // 2
        #     mask[:,:c,:] = 0.0
        # elif config.maskType == 'full':
        #     mask = np.ones(self.image_shape)
        # elif config.maskType == 'grid':
        #     mask = np.zeros(self.image_shape)
        #     mask[::4,::4,:] = 1.0
        # elif config.maskType == 'lowres':
        #     lowres_mask = np.ones(self.lowres_shape)
        #     mask = np.zeros(self.image_shape)
        else:
            assert(False)

        for idx in xrange(0, batch_ids):
            first_batch_id = idx*self.batch_size
            last_batch_id = min((idx+1)*self.batch_size, number_of_images)
            cur_batch_size = last_batch_id-first_batch_id
            batch_files = config.imgs[first_batch_id:last_batch_id]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if cur_batch_size < self.batch_size:
                padSz = ((0, int(self.batch_size-cur_batch_size)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            number_of_rows = np.ceil(cur_batch_size/8)
            number_of_cols = min(8, cur_batch_size)

            # mask!
            masked_images = np.multiply(batch_images, mask)
            save_images(masked_images[:cur_batch_size,:,:,:], [number_of_rows, number_of_cols],
                        os.path.join(config.outDir, 'masked.png'))
            save_images(batch_images[:cur_batch_size,:,:,:], [number_of_rows, number_of_cols],
                        os.path.join(config.outDir, 'before.png'))

            # setting up standard ADAM optimization!
            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            m = 0
            v = 0

            # number of Adam optimizations! (niter ~= 3000)
            for i in xrange(config.nIter):
                fd = {
                    self.z: zhats,
                    self.mask: mask,
                    self.lowres_mask: lowres_mask,
                    self.images: batch_images,
                    self.training_bool: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G, self.lowres_G]
                loss, g, G_imgs, lowres_G_imgs = self.sess.run(run, feed_dict=fd)

                # Save image, print Losses every outInterval steps
                if i % config.outInterval == 0:
                    print("Losses: ", i, np.mean(loss[0:cur_batch_size]))
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    number_of_rows = np.ceil(cur_batch_size/8)
                    number_of_cols = min(8, cur_batch_size)
                    save_images(G_imgs[:cur_batch_size,:,:,:], [number_of_rows,number_of_cols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
                    completed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    save_images(completed[:cur_batch_size,:,:,:], [number_of_rows,number_of_cols], imgName)

                if config.approach == 'adam':
                    # Optimize single completion with Adam
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                    v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - config.beta1 ** (i + 1))
                    v_hat = v / (1 - config.beta2 ** (i + 1))
                    zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                    zhats = np.clip(zhats, -1, 1)
                else:
                    # wrong default value
                    assert(False)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # Discriminator is four convolutional layers (RELU activation)
            hidden_layer_0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            hidden_layer_1 = lrelu(self.d_bns[0](conv2d(hidden_layer_0, self.df_dim*2, name='d_h1_conv'), self.training_bool))
            hidden_layer_2 = lrelu(self.d_bns[1](conv2d(hidden_layer_1, self.df_dim*4, name='d_h2_conv'), self.training_bool))
            hidden_layer_3 = lrelu(self.d_bns[2](conv2d(hidden_layer_2, self.df_dim*8, name='d_h3_conv'), self.training_bool))
            hidden_layer_4 = linear(tf.reshape(hidden_layer_3, [-1, 8192]), 1, 'd_h4_lin') # 64*64*2

            return tf.nn.sigmoid(hidden_layer_4), hidden_layer_4

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

            # TODO: Nicer iteration pattern here. #readability
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.training_bool))

            i = 1 # Iteration number.
            depth_mul = 8  # Depth decreases as spatial component increases.
            size = 8  # Size increases as depth decreases.

            # 4 convolutional layers as well..for 64x64
            while size < self.image_size:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i], _, _ = conv2d_transpose(hs[i-1],
                    [self.batch_size, size, size, self.gf_dim*depth_mul], name=name, with_w=True)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.training_bool))

                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i - 1],
                [self.batch_size, size, size, 3], name=name, with_w=True)

            return tf.nn.tanh(hs[i])

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Checking to see if checkpoints exist!!!...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
