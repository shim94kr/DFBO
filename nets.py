from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.5
FEATURE_NUM = 4
SOURCE_NUM = 4

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def pose_motion_net(tgt_image, src_image_stack, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    num_source = int(src_image_stack.get_shape()[3].value//3)
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1') # B * 64 * 208 * 16
                cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2') # B * 32 * 104 * 32
                cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3') # B * 16 * 52 * 64
                cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4') # B * 8 * 26 * 128
                cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5') # B * 4 * 13 * 256
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6') # B * 2 * 7 * 256
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7') # B * 1 * 4 * 256
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred1', # B * 1 * 4 * 24
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2]) # B * 24
                pose_final = 0.1 * tf.reshape(pose_avg, [-1, num_source, 6])
            with tf.variable_scope('shift'):
                upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                shift4_prev = DISP_SCALING * slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='shift4',
                                    normalizer_fn=None, activation_fn=tf.nn.tanh)
                shift4 = tf.sign(shift4_prev) * MIN_DISP + shift4_prev
                shift4_up = tf.image.resize_bilinear(shift4, [np.int(H / 4), np.int(W / 4)])  # B * H * W * 2

                upcnv3 = slim.conv2d_transpose(upcnv4, 64, [3, 3], stride=2, scope='upcnv3')
                shift3_in = tf.concat([shift4_up, upcnv3], axis=3)
                shift3_prev = DISP_SCALING * slim.conv2d(shift3_in, num_source * 2, [3, 3], stride=1, scope='shift3',
                                    normalizer_fn=None, activation_fn=tf.nn.tanh)
                shift3 = tf.sign(shift3_prev) * MIN_DISP * 2 + shift3_prev
                shift3_up = tf.image.resize_bilinear(shift3, [np.int(H / 2), np.int(W / 2)])  # B * H * W * 2

                upcnv2 = slim.conv2d_transpose(shift3, 32, [5, 5], stride=2, scope='upcnv2')
                shift2_in = tf.concat([shift3_up, upcnv2], axis=3)
                shift2_prev = DISP_SCALING * slim.conv2d(shift2_in, num_source * 2, [5, 5], stride=1, scope='shift2',
                                    normalizer_fn=None, activation_fn=tf.nn.tanh)
                shift2 = tf.sign(shift2_prev) * MIN_DISP * 4 + shift2_prev
                shift2_up = tf.image.resize_bilinear(shift2, [np.int(H), np.int(W)])  # B * H * W * 2

                upcnv1 = slim.conv2d_transpose(upcnv2, 16, [7, 7], stride=2, scope='upcnv1')
                shift1_in = tf.concat([shift2_up, upcnv1], axis = 3)
                shift1_prev = DISP_SCALING * slim.conv2d(shift1_in, num_source * 2, [7, 7], stride=1, scope='shift1',
                                    normalizer_fn=None, activation_fn=tf.nn.tanh)
                shift1 = tf.sign(shift1_prev) * MIN_DISP * 8 + shift1_prev
        end_points = utils.convert_collection_to_dict(end_points_collection)
        #return pose_final, end_points
        return pose_final, [shift1, shift2, shift3, shift4], end_points


def vgg_extractor(tgt_image, reuse=False):
    with tf.variable_scope('vgg_16', reuse=reuse) as t:
        conv1_1 = slim.conv2d(tgt_image, 64, [3, 3], scope='conv1/conv1_1')
        conv1_2 = slim.conv2d(conv1_1, 64, [3, 3], scope='conv1/conv1_2')
        pool1 = slim.max_pool2d(conv1_2, [2, 2], scope='pool1')

        conv2_1 = slim.conv2d(pool1, 128, [3, 3], scope='conv2/conv2_1')
        conv2_2 = slim.conv2d(conv2_1, 128, [3, 3], scope='conv2/conv2_2')
        pool2 = slim.max_pool2d(conv2_2, [2, 2], scope='pool2')

        conv3_1 = slim.conv2d(pool2, 256, [3, 3], scope='conv3/conv3_1')
        conv3_2 = slim.conv2d(conv3_1, 256, [3, 3], scope='conv3/conv3_2')

        pool3 = slim.max_pool2d(conv3_2, [2, 2], scope='pool3')

        conv4_1 = slim.conv2d(pool3, 512, [3, 3], scope='conv4/conv4_1')
        conv4_2 = slim.conv2d(conv4_1, 512, [3, 3], scope='conv4/conv4_2')

        normal_feat1 = tf.nn.l2_normalize(conv1_2, 3)  # B * H * W * feat_dim
        normal_feat2 = tf.nn.l2_normalize(conv2_2, 3)  # B * H * W * feat_dim
        normal_feat3 = tf.nn.l2_normalize(conv3_2, 3)  # B * H * W * feat_dim
        normal_feat4 = tf.nn.l2_normalize(conv4_2, 3)  # B * H * W * feat_dim

        return [normal_feat1, normal_feat2, normal_feat3, normal_feat4]


def mask_extractor (tgt_feature_map, num_source = SOURCE_NUM, do_exp=False, reuse=False):
    H = tgt_feature_map[0].get_shape()[1].value
    W = tgt_feature_map[0].get_shape()[2].value
    with slim.arg_scope([slim.conv2d],
                        normalizer_fn=None,
                        activation_fn=None):
        if do_exp:
            with tf.variable_scope('exp', reuse=reuse):
                mask4 = tf.sigmoid(slim.conv2d(tgt_feature_map[3], 1, [1, 1], scope = 'mask4'))
                mask4_up = tf.image.resize_bilinear(mask4, [np.int(H / 4), np.int(W / 4)])  # B * H * W * 2

                mask3_in = tf.concat([mask4_up, tgt_feature_map[2]], axis = 3)
                mask3 = tf.sigmoid(slim.conv2d(mask3_in, 1, [1, 1], scope = 'mask3'))
                mask3_up = tf.image.resize_bilinear(mask3, [np.int(H / 2), np.int(W / 2)])  # B * H * W * 2

                mask2_in = tf.concat([mask3_up, tgt_feature_map[1]], axis = 3)
                mask2 = tf.sigmoid(slim.conv2d(mask2_in, 1, [1, 1], scope = 'mask2'))
                mask2_up = tf.image.resize_bilinear(mask2, [np.int(H), np.int(W)])  # B * H * W * 2

                mask1_in = tf.concat([mask2_up, tgt_feature_map[0]], axis = 3)
                mask1 = tf.sigmoid(slim.conv2d(mask1_in, 1, [1, 1], scope = 'mask1'))
        else:
            mask1 = None
            mask2 = None
            mask3 = None
            mask4 = None
    return [mask1, mask2, mask3, mask4]


def make_grid(batch_size, H, W, feat_num = FEATURE_NUM):
    rowx1 = tf.range(W)  # W
    rowxH = tf.cast(tf.reshape(tf.tile(rowx1, [H]), [1, H, W]), tf.float32)  # 1 * H * W
    colx1 = tf.expand_dims(tf.range(H), 1)  # H
    colxW = tf.cast(tf.reshape(tf.tile(colx1, [1, W]), [1, H, W]), tf.float32)  # 1 * H * W
    ones_cnst = tf.ones(shape=[1, H, W])
    grid = tf.tile(tf.stack([rowxH, colxW, ones_cnst], axis=3), [batch_size, 1, 1, 1]) # B * H * W * 3

    return grid

def bilinear_sampler(imgs, coords):
  """Construct a new image by bilinear sampling from the input image.

  Points falling outside the source image boundary have value 0.

  Args:
    imgs: source image to be sampled from [batch, height_s, width_s, channels]
    coords: coordinates of source pixels to sample from [batch, height_t,
      width_t, 2]. height_t/width_t correspond to the dimensions of the output
      image (don't need to be the same as height_s/width_s). The two channels
      correspond to x and y coordinates respectively.
  Returns:
    A new sampled image [batch, height_t, width_t, channels]
  """
  def _repeat(x, n_repeats):
    rep = tf.transpose(
        tf.expand_dims(tf.ones(shape=tf.stack([
            n_repeats,
        ])), 1), [1, 0])
    rep = tf.cast(rep, 'float32')
    x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
    return tf.reshape(x, [-1])

  with tf.name_scope('image_sampling'):
    coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
    inp_size = imgs.get_shape()
    coord_size = coords.get_shape()
    out_size = coords.get_shape().as_list()
    out_size[3] = imgs.get_shape().as_list()[3]

    coords_x = tf.cast(coords_x, 'float32')
    coords_y = tf.cast(coords_y, 'float32')

    x0 = tf.floor(coords_x)
    x1 = x0 + 1
    y0 = tf.floor(coords_y)
    y1 = y0 + 1

    y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
    x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
    zero = tf.zeros([1], dtype='float32')

    x0_safe = tf.clip_by_value(x0, zero, x_max)
    y0_safe = tf.clip_by_value(y0, zero, y_max)
    x1_safe = tf.clip_by_value(x1, zero, x_max)
    y1_safe = tf.clip_by_value(y1, zero, y_max)

    ## bilinear interp weights, with points outside the grid having weight 0
    # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
    # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
    # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
    # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe

    ## indices in the flat image to sample from
    dim2 = tf.cast(inp_size[2], 'float32')
    dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
    base = tf.reshape(
        _repeat(
            tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
            coord_size[1] * coord_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = tf.reshape(x0_safe + base_y0, [-1])
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    ## sample from imgs
    imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
    imgs_flat = tf.cast(imgs_flat, 'float32')
    im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
    im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
    im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
    im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = tf.add_n([
        w00 * im00, w01 * im01,
        w10 * im10, w11 * im11
    ])
    return output

def vgg_saver():
    #include_list = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2', \
    #                'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2']
    include_list = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2', \
                    'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2', 'vgg_16/conv4/conv4_1', 'vgg_16/conv4/conv4_2']
    #include_list = ['vgg_16/conv1/conv1_1', 'vgg_16/conv1/conv1_2', 'vgg_16/conv2/conv2_1', 'vgg_16/conv2/conv2_2', \
    #                'vgg_16/conv3/conv3_1', 'vgg_16/conv3/conv3_2', 'vgg_16/conv4/conv4_1', 'vgg_16/conv4/conv4_2'
    #                'vgg_16/conv5/conv5_1', 'vgg_16/conv5/conv5_2', 'vgg_16/conv5/conv5_3']
    variables = slim.get_variables_to_restore(include = include_list)
    variables_to_restore = [v for v in variables]
    saver = tf.train.Saver(variables_to_restore)

    return saver

def softmax(target, axis, name=None):
  with tf.name_scope(name, 'softmax', values=[target]):
    max_axis = tf.reduce_max(target, axis, keep_dims=True)
    target_exp = tf.exp(target-max_axis)
    normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
    softmax = target_exp / normalize
    return softmax