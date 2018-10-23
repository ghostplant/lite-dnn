#!/usr/bin/env python3
'''
[Example]

# HOROVOD_GPU_ALLREDUCE=NCCL pip3 install --no-cache-dir --upgrade horovod
# g++ generator.cpp -fPIC -O3 -std=c++14 -shared -lpthread -lcuda -lcudart -lopencv_core -lopencv_highgui -lopencv_imgproc -I/usr/local/cuda/include -L/usr/local/cuda/lib64 && mpiexec -H 192.168.2.130,192.168.2.130,192.168.2.130,192.168.2.130,192.168.2.131,192.168.2.131,192.168.2.131,192.168.2.131,192.168.2.132,192.168.2.132,192.168.2.132,192.168.2.132  --mca oob_tcp_if_include enp216s0 --mca btl_tcp_if_include enp216s0 -x NCCL_SOCKET_IFNAME=enp216s0     --allow-run-as-root --map-by slot --bind-to none -x LD_PRELOAD=`pwd`/a.out -x NCCL_DEBUG=INFO -x GPUAAS_DATASET=/var/lib/docker/imagenet/train ./hvd-model.py
'''

import time, os, sys

batch_size, depth, height, width = 64, 3, 224, 224
os.environ['GPUAAS_BATCHSIZE'] = str(batch_size)
os.environ['GPUAAS_HEIGHT'] = str(height)
os.environ['GPUAAS_WIDTH'] = str(width)
os.environ['GPUAAS_BATCHSIZE'] = str(batch_size)
os.environ['GPUAAS_FORMAT'] = 'NCHW'

import tensorflow as tf
import numpy as np
import horovod.tensorflow as hvd

hvd.init()
device_rank = hvd.rank()
device_name = 'GPU: %%%dd/%%d' % len(str(hvd.size())) % (device_rank + 1, hvd.size())

global_step, query_freq = 1000000, 200


def create_dataset(data_dir=None):
  if data_dir is not None and data_dir[-1] != '/':
    data_dir += '/'
  n_classes = 1001

  if 'GPUAAS_DATASET' in os.environ:
    print('Using low-level dataset on %s..' % device_name)
    with tf.device('/cpu:0'):
      os.environ['IMAGE_SIZE'] = str(np.product([batch_size, depth, height, width]))
      os.environ['LABEL_SIZE'] = str(np.product([batch_size, ]))
      ds = tf.data.Dataset.from_tensors((tf.fill((batch_size, depth, height, width), np.float32(np.NaN)), tf.fill((batch_size, ), np.int32(0x7fc00001))))
      ds = ds.cache().repeat().prefetch(1024)
      images, labels = ds.make_one_shot_iterator().get_next()
      X = tf.reshape(images, (-1, depth, height, width))
      Y = tf.reshape(labels, (-1, ))
    return X, Y, n_classes

  if not data_dir:
    print('Using synthetic dataset on %s..' % device_name)
    X = tf.random_uniform((batch_size, depth, height, width), dtype=tf.float32, seed=1)
    Y = tf.random_uniform((batch_size, ), maxval=n_classes, dtype=tf.int32, seed=2)
    return X, Y, n_classes

  print('Using real dataset `%s` on %s..' % (data_dir, device_name))

  def generator():
    import warnings
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import OrderedEnqueuer
    gen = ImageDataGenerator(
      data_format='channels_first',
      rescale=1./255,
      horizontal_flip=True,
      rotation_range=10.0,
      zoom_range=0.2,
      width_shift_range=0.1,
      height_shift_range=0.1,
      fill_mode='nearest').flow_from_directory(data_dir, target_size=(height, width), batch_size=batch_size, class_mode='sparse')
    assert(gen.num_classes <= n_classes)
    enqueuer = OrderedEnqueuer(gen, use_multiprocessing=False)
    enqueuer.start(workers=16)
    while True:
      batch_xs, batch_ys = next(enqueuer.get())
      yield batch_xs, batch_ys
    enqueuer.close() # Never reached

  ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.int32))
  ds = ds.repeat().prefetch(buffer_size=1024)
  images, labels = ds.make_one_shot_iterator().get_next()
  X = tf.reshape(images, (-1, depth, height, width))
  Y = tf.reshape(labels, (-1,))
  return X, Y, n_classes


X, Y, n_classes = create_dataset(sys.argv[1] if len(sys.argv) > 1 else None)

def create_imagenet_resnet50v1(X, Y, n_classes):
  print('Creating imagenet_resnet50v1 on %s..' % device_name)

  # Using external models
  from models.resnet_model import create_resnet50_v2_model
  model = create_resnet50_v2_model()
  X, _ = model.build_network(X, nclass=n_classes, image_depth=depth, data_format='NHWC', phase_train=True, fp16_vars=False)
  loss = tf.losses.sparse_softmax_cross_entropy(logits=X, labels=Y)
  accuracy_3 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(X, Y, 3), tf.float32))
  accuracy_5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(X, Y, 5), tf.float32))
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(Y, tf.int64), tf.argmax(X, 1)), tf.float32))
  return loss, [accuracy, accuracy_3, accuracy_5], tf.trainable_variables()

loss, [accuracy, accuracy_3, accuracy_5], weights = create_imagenet_resnet50v1(X, Y, n_classes)

lr = 0.001 * hvd.size()

opt = tf.train.RMSPropOptimizer(lr, decay=0.95, momentum=0.9)
# opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
# opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
# opt = tf.train.AdamOptimizer(learning_rate=lr)
# opt = tf.train.AdadeltaOptimizer(learning_rate=lr)
# opt = tf.train.AdagradOptimizer(learning_rate=lr)

grad_vars = opt.compute_gradients(loss)

for i in range(len(grad_vars)):
  grad, var = grad_vars[i]
  grad = hvd.allreduce(grad, average=True)
  grad_vars[i] = (grad, var)

# train_op = opt.apply_gradients(grad_vars)
train_op = hvd.DistributedOptimizer(opt).minimize(loss)


config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = str(hvd.local_rank())


checkpoint_file = './resnet50-weights.npy'

try:
  weights_data = np.load(checkpoint_file)
except Exception as e:
  weights_data = None

with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  if device_rank == 0:
    if weights_data is not None:
      print('Try using pre-trained weights.')
      assign_ops = []
      for symbolic_weight, host_weight in zip(weights, weights_data):
        assign_ops.append(tf.assign(symbolic_weight, host_weight))
      sess.run(assign_ops)
    else:
      print('Not using pre-trained weights.')
  sess.run(hvd.broadcast_global_variables(0))

  if 'EVAL' in os.environ:
    for k in range(global_step):
      out_acc = sess.run([accuracy, accuracy_3, accuracy_5])
      print('[%s] top_1 = %.2f%%, top_3 = %.2f%%, top_5 = %.2f%%' % (device_name, out_acc[0] * 1e2, out_acc[1] * 1e2, out_acc[2] * 1e2))
    exit(0)

  print('Launch Training on %s..' % device_name)
  last_time, last_k = time.time(), -1
  for k in range(global_step):
    sess.run(train_op)
    if k == 0 or (k + 1) % query_freq == 0:
      curr_time = time.time()
      out_loss, out_acc = sess.run([loss, accuracy])
      during = curr_time - last_time
      print('[%s] step = %d (batch = %d; %.2f images/sec): loss = %.4f, acc = %.2f%%, during = %.3fs' % (device_name, (k + 1),
        batch_size, batch_size * (k - last_k) / during, out_loss, out_acc * 1e2, during))
      if (k + 1) % 1000 == 0 or (k + 1) == global_step:
         weights_data = sess.run(weights)
         if device_rank == 0:
           print('Saving current weights on [%s]..' % device_name)
           np.save(checkpoint_file, weights_data)
      last_time, last_k = time.time(), k
