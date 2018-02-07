from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from SfMLearner import SfMLearner
import os

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints_wh_test_seq5/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate",0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("int_loss", 1.0, "Weight for epipolar constraint")
flags.DEFINE_float("matching_loss", 1.0, "Weight for matching loss")
flags.DEFINE_float("epipolar_loss", 1.0, "Weight for epipolar loss")
flags.DEFINE_float("mask_loss", 1.0, "Weight for mask_loss")
flags.DEFINE_float("loc_loss", 1.0, "Weight for loc_loss")
flags.DEFINE_float("explain_reg_weight", 0.01, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 5, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 2000000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
FLAGS = flags.FLAGS

def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)
    
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

#    with tf.device('/device:GPU:1'):
    sfm = SfMLearner()
    sfm.train(FLAGS)

if __name__ == '__main__':
    tf.app.run()
