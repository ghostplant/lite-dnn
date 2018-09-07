#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import warnings, time

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def u_rand(seed):
  seed = ((seed & 0x7fffffff) * 1103515245 + 12345) & 0x7fffffff
  return seed

batch_size = 64
steps = 500000

image_shapes = (None, 3, 224, 224)
label_shapes = (None, 2)

place_X = tf.placeholder(tf.float32, list(image_shapes))
place_Y = tf.placeholder(tf.float32, list(label_shapes))

def gen_fixed_random(shape, limit):
  # Fill random data that matches lite-model
  limit = limit * 2 / float(0x7fffffff)
  count = np.product(shape)
  arr = []
  seed = count
  cc = 0.5 * float(0x7fffffff)
  for i in range(count):
    seed = u_rand(seed)
    val = (seed - cc) * limit
    arr.append(np.float32(val))
  val = np.array(arr).reshape(shape).astype(np.float32)
  if len(shape) == 4:
    val = val.reshape([-1])
    stride = [shape[1], 1, shape[1] * shape[0], shape[2] * shape[1] * shape[0]]
    new_val = np.zeros(shape, dtype=np.float32)

    for i0 in range(shape[0]):
      for i1 in range(shape[1]):
        for i2 in range(shape[2]):
          for i3 in range(shape[3]):
            new_val[i0][i1][i2][i3] = val[i0 * stride[0] + i1 * stride[1] + i2 * stride[2] + i3 * stride[3]]
    val = new_val
  return val

def energy(val):
  val = val.reshape([-1])
  tot = 0
  for i in range(len(val)):
    tot += val[i] * val[i]
  return tot

def init_weight(shape):
  if len(shape) == 2:
    fan_in, fan_out = shape[0], shape[1]
  elif len(shape) == 4:
    fan_in, fan_out = shape[0] * shape[1] * shape[2], shape[0] * shape[1] * shape[3]
  limit = np.sqrt(6.0 / (fan_in + fan_out))
  return tf.Variable(tf.random_uniform(shape, -limit, limit))
  # return tf.Variable(gen_fixed_random(shape, limit))


def conv2d(X, filter, ksize, stride, padding):
  return tf.nn.conv2d(tf.pad(X, tf.constant([[0, 0], [0, 0], [padding, padding], [padding, padding]])), init_weight([ksize, ksize, int(X.shape[1]), filter]), strides=[1, 1, stride, stride], data_format='NCHW', padding='VALID')

def dense(X, out_channels):
  return tf.matmul(X, init_weight([int(X.shape[1]), out_channels])) + tf.Variable(tf.zeros([out_channels]))

def mpool2d(X, ksize, stride):
  return tf.nn.max_pool(X, ksize=[1, 1, ksize, ksize], strides=[1, 1, stride, stride], padding='VALID', data_format='NCHW')

def lrn(X):
  return tf.nn.lrn(X, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

def flatten(X):
  return tf.reshape(X, shape=[-1, np.product(X.shape[1:])])


def create_imagenet_alexnet(X):
  print('Creating imagenet_alexnet ..')
  X = conv2d(X, 96, 11, 4, 2)
  X = tf.nn.relu(X)
  X = lrn(X)
  X = mpool2d(X, 3, 2)
  X = conv2d(X, 256, 5, 1, 2)
  X = tf.nn.relu(X)
  X = lrn(X)
  X = mpool2d(X, 3, 2)
  X = conv2d(X, 384, 3, 1, 1)
  X = lrn(X)
  X = tf.nn.relu(X)
  X = conv2d(X, 256, 3, 1, 1)
  X = tf.nn.relu(X)
  X = lrn(X)
  X = mpool2d(X, 3, 2)
  X = flatten(X)

  X = dense(X, 4096)
  X = tf.nn.relu(X)
  X = tf.nn.dropout(X, 0.25)
  X = dense(X, 4096)
  X = tf.nn.relu(X)
  X = tf.nn.dropout(X, 0.25)
  X = dense(X, label_shapes[1])
  return X


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:
  X = create_imagenet_alexnet(place_X)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=place_Y))
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(X, 1), tf.argmax(place_Y, 1)), tf.float32))
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)

  sess.run(tf.global_variables_initializer())

  from keras.preprocessing.image import ImageDataGenerator
  from keras.utils import OrderedEnqueuer
  gen = ImageDataGenerator(
        data_format='channels_first',
        rescale=1./255,
        fill_mode='nearest').flow_from_directory('/tmp/dataset/catsdogs/train', target_size=image_shapes[2:], batch_size=batch_size)
  enqueuer = OrderedEnqueuer(gen, use_multiprocessing=True)
  enqueuer.start(workers=4)

  init_time = last_time = time.time()
  k = 1
  for batch_xs, batch_ys in enqueuer.get():
    sess.run(optimizer, feed_dict={place_X: batch_xs, place_Y: batch_ys})
    curr_time = time.time()
    if k % 100 == 0:
    # if curr_time >= last_time + 1.0:
      out_loss, out_acc = sess.run([loss, accuracy], feed_dict={place_X: batch_xs, place_Y: batch_ys})
      print('step = %d (batch = %d; %.2f images/sec): loss = %.4f, acc = %.1f%%, time = %.3fs' % (k, batch_size, batch_size * (k + 1) / (curr_time - init_time), out_loss, out_acc *1e2, curr_time - last_time))
      last_time = curr_time
    k = k + 1
    if k > steps:
      break
