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

def init_weight(shape):
  if len(shape) == 2:
    fan_in, fan_out = shape[0], shape[1]
  elif len(shape) == 4:
    fan_in, fan_out = shape[0] * shape[1] * shape[2], shape[0] * shape[1] * shape[3]
  limit = np.sqrt(6.0 / (fan_in + fan_out))
  return tf.Variable(tf.random_uniform(shape, -limit, limit))
  # return tf.Variable(gen_fixed_random(shape, limit))


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


def create_imagenet_alexnet(X, n_classes):
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
  X = dense(X, n_classes)
  return X


def create_imagenet_resnet50v1(X, n_classes):
  print('Creating imagenet_resnet50v1 ..')
  assert(int(X.shape[2]) == 224 and int(X.shape[3]) == 224 and int(X.shape[1]) == 3)
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
  X = dense(X, n_classes)
  return X


def create_dataset(data_dir, batch_size=64, height=224, width=224):
  # Prepare Dataset
  from keras.preprocessing.image import ImageDataGenerator
  from keras.utils import OrderedEnqueuer

  gen = ImageDataGenerator(data_format='channels_first', rescale=1./255, fill_mode='nearest').flow_from_directory(
                           data_dir + '/train', target_size=(height, width), batch_size=batch_size)
  val_gen = ImageDataGenerator(data_format='channels_first', rescale=1./255).flow_from_directory(
                               data_dir + '/validate', target_size=(height, width), batch_size=batch_size)

  if gen.num_classes != val_gen.num_classes:
    raise Exception("The number of train classes and validate classes must be equal.")

  enqueuer = OrderedEnqueuer(gen, use_multiprocessing=False)
  enqueuer.start(workers=8)
  val_enqueuer = OrderedEnqueuer(gen, use_multiprocessing=False)
  val_enqueuer.start(workers=1)
  return enqueuer, val_enqueuer, gen.num_classes, (3, height, width)


config = tf.ConfigProto()
config.gpu_options.allow_growth=True

batch_size, steps = 64, 10000
# data_dir = '/var/lib/docker/imagenet'
data_dir = '/tmp/dataset/catsdogs'

with tf.Session(config=config) as sess:
  enqueuer, val_enqueuer, n_classes, image_shape = create_dataset(data_dir=data_dir, batch_size=batch_size)
  place_X = tf.placeholder(tf.float32, (None,) + image_shape)
  place_Y = tf.placeholder(tf.float32, (None, n_classes))

  # X = create_imagenet_alexnet(place_X, n_classes)
  X = create_imagenet_resnet50v1(place_X, n_classes)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=place_Y))
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(X, 1), tf.argmax(place_Y, 1)), tf.float32))
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(loss)

  sess.run(tf.global_variables_initializer())

  session_ckpt = "/tmp/model.ckpt"
  saver = tf.train.Saver()
  try:
    saver.restore(sess, session_ckpt + "/session")
    print('Restore last session.')
  except:
    print('Create new session.')

  # Start Training
  init_time = last_time = time.time()
  for k in range(1, steps + 1):
    batch_xs, batch_ys = next(enqueuer.get())
    sess.run(optimizer, feed_dict={place_X: batch_xs, place_Y: batch_ys})
    curr_time = time.time()
    if k % 100 == 0 or k == 1:
      val_xs, val_ys = next(val_enqueuer.get())
      out_loss, out_acc = sess.run([loss, accuracy], feed_dict={place_X: batch_xs, place_Y: batch_ys})
      val_loss, val_acc = sess.run([loss, accuracy], feed_dict={place_X: val_xs, place_Y: val_ys})
      print('step = %d (batch = %d; %.2f images/sec): loss = %.4f, acc = %.1f%%, val_loss = %.4f, val_acc = %.1f%%, time = %.3fs' %
        (k, batch_size, batch_size * k / (curr_time - init_time), out_loss, out_acc * 1e2, val_loss, val_acc * 1e2, curr_time - last_time))
      last_time = curr_time
    if k % 1000 == 0 or k == steps:
      save_path = saver.save(sess, session_ckpt + "/session")
      print("Model saved in path: %s" % save_path)
