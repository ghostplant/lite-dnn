#!/usr/bin/env python3

import numpy as np
import time, os
import tensorflow as tf
import horovod.tensorflow as hvd

hvd.init()
tf_weights = dict()

def init_weight(shape):
  if len(shape) == 2:
    fan_in, fan_out = shape[0], shape[1]
  elif len(shape) == 4:
    fan_in, fan_out = shape[0] * shape[1] * shape[2], shape[0] * shape[1] * shape[3]
  limit = np.sqrt(6.0 / (fan_in + fan_out))
  scope = tf.get_default_graph().get_name_scope()
  if scope not in tf_weights:
    tf_weights[scope] = []
  seed = len(tf_weights[scope])
  var = tf.Variable(tf.random_uniform(shape, -limit, limit, seed=seed))
  tf_weights[scope].append(var)
  return var


def conv2d(X, filter, ksize, stride, padding=0):
  if padding != 0:
    X = tf.pad(X, tf.constant([[0, 0], [0, 0], [padding, padding], [padding, padding]]))
  return tf.nn.conv2d(X, init_weight([ksize, ksize, int(X.shape[1]), filter]), strides=[1, 1, stride, stride], data_format='NCHW', padding='VALID')

def dense(X, out_channels):
  return tf.matmul(X, init_weight([int(X.shape[1]), out_channels])) + tf.Variable(tf.zeros([out_channels]))

def mpool2d(X, ksize, stride):
  return tf.nn.max_pool(X, ksize=[1, 1, ksize, ksize], strides=[1, 1, stride, stride], padding='VALID', data_format='NCHW')

def apool2d(X, ksize, stride):
  return tf.nn.avg_pool(X, ksize=[1, 1, ksize, ksize], strides=[1, 1, stride, stride], padding='VALID', data_format='NCHW')

def lrn(X):
  return tf.nn.lrn(X, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def flatten(X):
  return tf.reshape(X, shape=[-1, np.product(X.shape[1:])])


batch_size, depth, height, width, n_classes = 64, 3, 224, 224, 2


device_name = 'GPU:%d/%d' % (hvd.rank(), hvd.size())


def create_imagenet_resnet50v1(X, Y, n_classes):
  print('Creating imagenet_resnet50v1 on `%s`..' % device_name)

  X = conv2d(X, 64, 7, 2, 3)
  X = tf.nn.relu(X)
  X = mpool2d(X, 3, 2)

  def bottleneck_block_v1(X, depth, depth_bottleneck, stride):
    shortcut = None
    if depth == int(X.shape[1]):
      if stride == 1:
        shortcut = X
      else:
        shortcut = mpool2d(X, 1, 2)
    else:
      shortcut = conv2d(X, depth, 1, stride)
      shortcut = tf.nn.relu(shortcut)
    output = conv2d(X, depth_bottleneck, 1, stride)
    output = tf.nn.relu(output)
    output = conv2d(output, depth_bottleneck, 3, 1, 1)
    output = tf.nn.relu(output)
    output = conv2d(output, depth, 1, 1)
    return tf.nn.relu(output + shortcut)

  layer_counts = [3, 4, 6, 3]
  for i in range(layer_counts[0]):
    X = bottleneck_block_v1(X, 256, 64, 1)
  for i in range(layer_counts[1]):
    X = bottleneck_block_v1(X, 512, 128, 2 if i == 0 else 1)
  for i in range(layer_counts[2]):
    X = bottleneck_block_v1(X, 1024, 256, 2 if i == 0 else 1)
  for i in range(layer_counts[3]):
    X = bottleneck_block_v1(X, 2048, 512, 2 if i == 0 else 1)

  X = apool2d(X, int(X.shape[2]), 1)
  X = flatten(X)

  denseY = tf.one_hot(Y, n_classes)
  X = dense(X, n_classes)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=denseY))
  return loss

train_ops, losses = [], []
grad_avg, opts = [], []
tower_grads, tower_vars = [], []

config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = str(hvd.local_rank())

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

with tf.Session(config=config) as sess:
  ngpus = 1 # len(get_available_gpus())

  for i in range(ngpus):
    with tf.name_scope('gpu_%d' % i), tf.device('/gpu:%d' % i):
      image_shape = [batch_size, depth, height, width]
      label_shape = [batch_size]
      X = tf.random_uniform(image_shape, dtype=tf.float32, seed=hvd.rank())
      Y = tf.random_uniform(label_shape, maxval=n_classes, dtype=tf.int32, seed=hvd.rank())
      '''
      with tf.device('/cpu:0'):
        os.environ['IMAGE_BATCH'] = str(np.product(image_shape))
        os.environ['LABEL_BATCH'] = str(np.product(label_shape))
        ds = tf.data.Dataset.from_tensors((tf.fill(image_shape, np.float32(np.NaN)), tf.fill(label_shape, np.int32(0x7fc00001))))
        images, labels = ds.repeat().make_one_shot_iterator().get_next()
        X = tf.reshape(images, image_shape)
        Y = tf.reshape(labels, label_shape) '''

      lr = 0.01 / ngpus
      opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
      # opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
      # opt = tf.train.AdamOptimizer(learning_rate=lr)
      # opt = tf.train.AdadeltaOptimizer(learning_rate=lr)
      # opt = tf.train.AdagradOptimizer(learning_rate=lr)

      loss = create_imagenet_resnet50v1(X, Y, n_classes)
      grad_gv = opt.compute_gradients(loss, tf_weights[tf.get_default_graph().get_name_scope()])
      grad_g = [hvd.allreduce(g, average=True) for g, _ in grad_gv]
      grad_v = [v for _, v in grad_gv]
      tower_grads.append(grad_g)
      tower_vars.append(grad_v)
      losses.append(loss)
      opts.append(opt)

  if ngpus == 1:
    it = next(zip(*tower_grads))
    grad_avg = [it]
  else:
    for it in zip(*tower_grads):
      all_sum = tf.contrib.nccl.all_sum(it)
      grad_avg.append(all_sum)

  for i, it in enumerate(zip(*grad_avg)):
    with tf.name_scope('gpu_%d' % i), tf.device('/gpu:%d' % i):
      grad = zip(list(it), tower_vars[i])
      train_ops.append(opts[i].apply_gradients(grad))

  mean_loss = tf.reduce_sum(losses) / ngpus

  sess.run(tf.global_variables_initializer())

  # Synchronize initial weights
  print('Warnup Variables on `%s` ..' % device_name)
  sess.run(train_ops)

  print('Launch Training on `%s` ..' % device_name)
  freq = 100
  init_time = last_time = time.time()
  for k in range(100000):
    sess.run(train_ops)
    if (k + 1) % freq == 0:
      curr_time = time.time()
      print('[%s] step = %d (batch = %d; %.2f images/sec): loss = %.4f, during = %.3fs' % (device_name, (k + 1),
        batch_size * ngpus, batch_size * ngpus * (k + 1) / (curr_time - init_time), sess.run(mean_loss), curr_time - last_time))

      last_time = time.time()
