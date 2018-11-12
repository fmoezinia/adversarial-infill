#!/usr/bin/env python3

# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/main.py
#   + License: MIT
# [2016-08-05] Modifications for Inpainting: Brandon Amos (http://bamos.github.io)
#   + License: MIT

import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("lamda", 0.1, "Learning rate")
flags.DEFINE_integer("train_size", 1000000000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("sample_size", 64, "The Sample Size")
flags.DEFINE_integer("image_size", 64, "The size of image to use")
flags.DEFINE_integer("lowres", 8, "Low resolution image/mask shrink factor")
flags.DEFINE_integer("z_dim", 100, "Z dimension")
flags.DEFINE_integer("gf_dim", 64, "First Conv Layer Generator Dimension")
flags.DEFINE_integer("df_dim", 64, "First Conv Layer Discriminator Dimension") # Convolution is when you take a matrix and stride it around taking sum of 'square dot products'
flags.DEFINE_integer("gfc_dim", 1024, "First FC Layer Generator Dimension")
flags.DEFINE_integer("dfc_dim", 1024, "First FC Layer Discriminator Dimension")
flags.DEFINE_integer("c_dim", 3, "Colour Dimension") # RGB
flags.DEFINE_string("dataset", "your-dataset/aligned", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")

FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess,
                  image_size=FLAGS.image_size,
                  batch_size=FLAGS.batch_size,
                  sample_size=FLAGS.sample_size,
                  lowres=FLAGS.lowres,
                  z_dim=FLAGS.z_dim,
                  gf_dim=FLAGS.gf_dim,
                  df_dim=FLAGS.df_dim,
                  gfc_dim=FLAGS.gfc_dim,
                  dfc_dim=FLAGS.dfc_dim,
                  c_dim=FLAGS.c_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  lamda=FLAGS.lamda,
                  )

    dcgan.train(FLAGS)
