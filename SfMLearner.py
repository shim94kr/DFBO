11from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from nets import *
from utils import *
from tensorflow.python import debug as tf_debug

class SfMLearner(object):
    def __init__(self):
        pass
    
    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):
            tgt_image, src_image_stack, intrinsics = loader.load_train_batch()
            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)

        with tf.name_scope("feat_ext_net"):
            tgt_feature_map = vgg_extractor(tgt_image, reuse=False) # batch * H * W * feat_dim
            for i in range(opt.num_source):
                src_feature_map = vgg_extractor(src_image_stack[:,:,:,3*i:3*(i+1)], reuse=True)
                if opt.explain_reg_weight > 0:
                    pred_exp_logit_1 = mask_extractor(src_feature_map, do_exp=(opt.explain_reg_weight > 0), reuse=tf.AUTO_REUSE)
                for j in range(opt.num_scales) :
                    src_feature_map[j] = tf.expand_dims(src_feature_map[j], 0)
                    if opt.explain_reg_weight > 0:
                        pred_exp_logit_1[j] = tf.expand_dims(pred_exp_logit_1[j], 0)
                if i==0 :
                    src_feature_map_stack = src_feature_map
                    if opt.explain_reg_weight > 0:
                        pred_exp_logits = pred_exp_logit_1
                else :
                    for j in range(opt.num_scales):
                        src_feature_map_stack[j] = tf.concat([src_feature_map_stack[j], src_feature_map[j]], axis=0)
                        if opt.explain_reg_weight > 0:
                            pred_exp_logits[j] = tf.concat([pred_exp_logits[j], pred_exp_logit_1[j]], axis=0)


        with tf.name_scope("pose_and_explainability_prediction"):
            # pred_poses, pred_exp_logits, pose_exp_net_endpoints = \
            pred_poses, motion_map, pose_exp_net_endpoints = \
                pose_motion_net(tgt_image,
                             src_image_stack,
                             is_training=True)

        with tf.name_scope("compute_loss"):
            epipolar_loss_debug = 0
            matching_loss_debug = 0
            exp_loss = 0
            integrated_loss = 0
            tgt_image_all = []
            src_image_stack_all = []
            exp_mask_stack_all = []

            for s in range(opt.num_scales):
                if opt.explain_reg_weight > 0:
                    ref_exp_mask = self.get_reference_explain_mask(s)
                batch_size, H, W, feat_dim = tgt_feature_map[s].get_shape().as_list()
                grid_src = make_grid(batch_size, H, W)
                grid_src_rs = tf.reshape(grid_src, [batch_size, H*W, 3])

                curr_tgt_image = tf.image.resize_area(tgt_image,
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])
                curr_src_image_stack = tf.image.resize_area(src_image_stack,
                    [int(opt.img_height/(2**s)), int(opt.img_width/(2**s))])

                tgt_feature_norm = tf.nn.l2_normalize(tgt_feature_map[s], 3)
                y_max = tf.cast(H - 1, 'float32')
                x_max = tf.cast(W - 1, 'float32')
                zero = tf.zeros([1], dtype='float32')
                for i in range(opt.num_source):
                    if opt.explain_reg_weight > 0:
                        curr_exp = tf.sigmoid(pred_exp_logits[s][i])
                        exp_loss += opt.explain_reg_weight * \
                            self.compute_exp_reg_loss(curr_exp,
                                                      ref_exp_mask)

                    rowxH = grid_src[:, :, :, 0] + motion_map[s][:, :, :, i*2]
                    colxW = grid_src[:, :, :, 1] + motion_map[s][:, :, :, i*2+1]
                    rowxH_safe = tf.clip_by_value(rowxH, zero, x_max)
                    colxW_safe = tf.clip_by_value(colxW, zero, y_max)
                    grid_tgt_from_src = tf.stack([rowxH_safe, colxW_safe], axis=3)  # B * H * W * 2
                    src_feature_norm = tf.nn.l2_normalize(src_feature_map_stack[s][i], 3)  # B * H * W * feat_dim
                    tgt_feature_from_src = bilinear_sampler(tgt_feature_norm, grid_tgt_from_src)
                    matching_error = tf.abs(tgt_feature_from_src - src_feature_norm) # s, batch * feat_num * feat_dim -> B * feat_num

                    if opt.explain_reg_weight > 0:
                        matching_loss = tf.reduce_mean(matching_error * tf.expand_dims(curr_exp[:,:,:,0], -1), axis = 3)
                    else:
                        matching_loss = tf.reduce_mean(matching_error, axis = 3)
                    matching_loss_debug += tf.reduce_mean(matching_loss)

                    rot1 = pose_vec2mat(pred_poses[:, i, :])[:, 0, :3]  # B * 3
                    rot2 = pose_vec2mat(pred_poses[:, i, :])[:, 1, :3]  # B * 3
                    rot3 = pose_vec2mat(pred_poses[:, i, :])[:, 2, :3]  # B * 3
                    trans = pose_vec2mat(pred_poses[:, i, :])[:, :3, 3]  # B * 3
                    essential_matrix  = tf.stack([tf.cross(rot1, trans), tf.cross(rot2, trans), tf.cross(rot3, trans)], axis=1)  # B * 3 * 3
                    inv_intrinsic = tf.matrix_inverse(intrinsics[:, s, :, :])
                    fundamental_matrix = tf.matmul(tf.matmul(tf.transpose(inv_intrinsic, [0, 2, 1]), essential_matrix), inv_intrinsic) # B * 3 * 3

                    grid_tgt_cct1 = tf.concat([grid_tgt_from_src, tf.ones([batch_size, H, W, 1])], axis = 3)
                    grid_tgt_rs = tf.reshape(grid_tgt_cct1, [batch_size, H*W, 3])
                    epiline1 = tf.abs(tf.matmul(grid_src_rs, fundamental_matrix)) # B * HW * 3
                    epipolar_error = epiline1 * grid_tgt_rs # B * HW * 3
                    epipolar_error_rs = tf.reshape(epipolar_error, [batch_size, H, W, 3])

                    if opt.explain_reg_weight > 0:
                        epipolar_loss = tf.reduce_mean(epipolar_error_rs * tf.expand_dims(curr_exp[:,:,:,0], -1), axis = 3)
                    else:
                        epipolar_loss = tf.reduce_mean(epipolar_error_rs, axis = 3)

                    epipolar_loss_debug += tf.reduce_mean(epipolar_loss)
                    integrated_loss += tf.reduce_mean(matching_loss*s + epipolar_loss*(2**(4-s)))

                    if opt.explain_reg_weight > 0:
                        if i==0:
                            exp_mask_stack = tf.expand_dims(curr_exp[:,:,:,0], -1)
                        else:
                            exp_mask_stack = tf.concat([exp_mask_stack, tf.expand_dims(curr_exp[:,:,:,0], -1)], axis=3)
                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                if opt.explain_reg_weight > 0:
                    exp_mask_stack_all.append(exp_mask_stack)

        total_loss = opt.int_loss * integrated_loss + exp_loss
        #total_loss = opt.matching_loss * matching_loss_debug + opt.epipolar_loss * epipolar_loss_debug

        with tf.name_scope("train_op"):
            #train_vars = [var for var in tf.trainable_variables() if not 'vgg_16' in var.name]
            #train_vars = [var for var in tf.trainable_variables() if not 'attention' in var.name]
            train_vars = [var for var in tf.trainable_variables()]
            optim = tf.train.GradientDescentOptimizer(opt.learning_rate)
            #optim = tf.train.AdamOptimizer(opt.learning_rate, opt.beta1)
            self.grads_and_vars = optim.compute_gradients(total_loss,
                                                         var_list=train_vars)
            self.train_op = optim.apply_gradients(self.grads_and_vars)
            self.train_op = slim.learning.create_train_op(total_loss, optim)
            self.global_step = tf.Variable(0, 
                                           name='global_step', 
                                           trainable=False)
            self.incr_global_step = tf.assign(self.global_step, 
                                              self.global_step+1)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_poses = pred_poses
        self.steps_per_epoch = loader.steps_per_epoch
        self.total_loss = total_loss
        self.exp_loss = exp_loss
        self.motion_map = motion_map
        self.epipolar_loss_debug = epipolar_loss_debug
        self.matching_loss_debug = matching_loss_debug
        self.integrated_loss = integrated_loss
        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.exp_mask_stack_all = exp_mask_stack_all

    def compute_exp_reg_loss(self, pred, ref):
        pred = tf.stack([1-pred, pred], axis=3)
        l = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(ref, [-1, 2]),
            logits=tf.reshape(pred, [-1, 2]))
        return tf.reduce_mean(l)

    def get_reference_explain_mask(self, downscaling):
        opt = self.opt
        tmp = np.array([0,1])
        ref_exp_mask = np.tile(tmp, 
                               (opt.batch_size, 
                                int(opt.img_height/(2**downscaling)), 
                                int(opt.img_width/(2**downscaling)), 
                                1))
        ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
        return ref_exp_mask

    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("integrated_loss", self.integrated_loss)
        tf.summary.scalar("exp_loss", self.exp_loss)
        tf.summary.scalar("epipolar_loss_debug", self.epipolar_loss_debug)
        tf.summary.scalar("matching_loss_debug", self.matching_loss_debug)
        tf.summary.histogram("tx", self.pred_poses[:,:,0])
        tf.summary.histogram("ty", self.pred_poses[:,:,1])
        tf.summary.histogram("tz", self.pred_poses[:,:,2])
        tf.summary.histogram("rx", self.pred_poses[:,:,3])
        tf.summary.histogram("ry", self.pred_poses[:,:,4])
        tf.summary.histogram("rz", self.pred_poses[:,:,5])
        for s in range(opt.num_scales):
            tf.summary.image('scale%d_target_image' % s, self.deprocess_image(self.tgt_image_all[s][:1, :, :, :])) # B * H * W * 3
            for i in range(opt.num_source):
                tf.summary.histogram("motion_x_%d_%d" % (s, i), self.motion_map[s][:, :, :, i*2])
                tf.summary.histogram("motion_y_%d_%d" % (s, i), self.motion_map[s][:, :, :, i*2+1])

            for i in range(opt.num_source):
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i),
                    self.deprocess_image(self.src_image_stack_all[s][:1, :, :, i*3:(i+1)*3]))
                if opt.explain_reg_weight > 0:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i),
                        tf.expand_dims(self.exp_mask_stack_all[s][:1,:,:,i], -1))

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)
        for grad, var in self.grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        opt.num_scales = 4
        self.opt = opt
        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                            for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.model_variables()] + \
                                    [self.global_step],
                                     max_to_keep=10)
        self.vgg_saver = vgg_saver()
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir, 
                                 save_summaries_secs=0, 
                                 saver=None)
        vgg_model_path = './vgg_16.ckpt'
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print ('load vgg model')
            self.vgg_saver.restore(sess, vgg_model_path)
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                    #checkpoint = './checkpoint107/model-43884'
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, opt.max_steps):
                #print ("** %d", step)
                #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step,
                    #"vgg_feature_map": self.vgg_feature_map
                }

                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op

                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f" \
                            % (train_epoch, train_step, self.steps_per_epoch, \
                                (time.time() - start_time)/opt.summary_freq, 
                                results["loss"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size, 
            self.img_height, self.img_width * self.seq_length, 3], 
            name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_motion_net(
                tgt_image, src_image_stack, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self, 
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs:inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess, 
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)

