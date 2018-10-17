#!/usr/bin/env python3
'''
[Example]

# HOROVOD_GPU_ALLREDUCE=NCCL pip3 install --no-cache-dir --upgrade horovod
# g++ main.cpp -fPIC -shared -lpthread -lcuda -lcudart -I/usr/local/cuda/include -L/usr/local/cuda/lib64 && mpiexec -H 192.168.2.130,192.168.2.130,192.168.2.130,192.168.2.130,192.168.2.131,192.168.2.131,192.168.2.131,192.168.2.131,192.168.2.132,192.168.2.132,192.168.2.132,192.168.2.132 \
    --mca oob_tcp_if_include enp216s0 --mca btl_tcp_if_include enp216s0 -x NCCL_SOCKET_IFNAME=enp216s0 \
    --allow-run-as-root --map-by slot --bind-to none -x LD_PRELOAD=`pwd`/a.out -x NCCL_DEBUG=INFO ./hvd-model.py # /usr/imagenet/flower_photos
'''

import tensorflow as tf
import numpy as np
import time, os, sys
import horovod.tensorflow as hvd

hvd.init()
device_name = 'GPU: %%%dd/%%d' % len(str(hvd.size())) % (hvd.rank() + 1, hvd.size())

batch_size, depth, height, width = 64, 3, 224, 224
global_step, query_freq = 100000, 100


def create_dataset(synthetic=False):
  if len(sys.argv) <= 1:
    print('Using synthetic dataset on %s..' % device_name)
    n_classes = 1001
    '''X = tf.random_uniform((batch_size, depth, height, width), dtype=tf.float32, seed=1)
    Y = tf.random_uniform((batch_size, n_classes), maxval=1.0/n_classes, dtype=tf.float32, seed=1)
    return X, Y, n_classes'''
    with tf.device('/cpu:0'):
      X = tf.fill((batch_size, depth, height, width), np.float32(np.NaN))
      Y = tf.fill((batch_size, n_classes), np.float32(np.NaN))
      os.environ['IMAGE_SIZE'] = str(np.product([batch_size, depth, height, width]))
      os.environ['LABEL_SIZE'] = str(np.product([batch_size, n_classes]))
    ds = tf.data.Dataset.from_tensors((X, Y)).repeat()
    images, labels = ds.make_one_shot_iterator().get_next()
    X = tf.reshape(images, (-1, depth, height, width))
    Y = tf.reshape(labels, (-1, n_classes))
    return X, Y, n_classes

  data_dir = sys.argv[1]
  print('Using real dataset `%s` on %s..' % (data_dir, device_name))

  if data_dir[-1] != '/':
    data_dir += '/'
  n_classes = 0
  for subdir in os.listdir(data_dir):
    path = os.path.join(data_dir, subdir)
    if os.path.isdir(path):
      n_classes = n_classes + 1

  def generator():
    import warnings
    def warn(*args, **kwargs):
        pass
    warnings.warn = warn

    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import OrderedEnqueuer
    gen = ImageDataGenerator(data_format='channels_first', rescale=1./255, fill_mode='nearest').flow_from_directory(
                             data_dir, target_size=(height, width), batch_size=batch_size)
    assert(n_classes == gen.num_classes)
    enqueuer = OrderedEnqueuer(gen, use_multiprocessing=False)
    enqueuer.start(workers=8)
    while True:
      batch_xs, batch_ys = next(enqueuer.get())
      yield batch_xs, batch_ys
    enqueuer.close() # Never reached

  ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
  ds = ds.prefetch(buffer_size=1024)
  images, labels = ds.make_one_shot_iterator().get_next()
  X = tf.reshape(images, (-1, depth, height, width))
  Y = tf.reshape(labels, (-1, n_classes))
  return X, Y, n_classes

X, Y, n_classes = create_dataset()

def create_imagenet_resnet50v1(X, Y, n_classes):
  print('Creating imagenet_resnet50v1 on %s..' % device_name)

  # Using external models
  from models.resnet_model import create_resnet50_model
  model = create_resnet50_model()
  X, _ = model.build_network(X, nclass=n_classes, image_depth=depth, data_format='NHWC', phase_train=True, fp16_vars=False)

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=Y))
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(Y, 1), tf.argmax(X, 1)), tf.float32))
  return loss, accuracy, tf.trainable_variables()

loss, accuracy, weights = create_imagenet_resnet50v1(X, Y, n_classes)

lr = 0.01
# opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
# opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
# opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)
# opt = tf.train.AdamOptimizer(learning_rate=lr)
# opt = tf.train.AdadeltaOptimizer(learning_rate=lr)
opt = tf.train.AdagradOptimizer(learning_rate=lr)

grad_vars = opt.compute_gradients(loss)

for i in range(len(grad_vars)):
  grad, var = grad_vars[i]
  grad = hvd.allreduce(grad, average=True)
  grad_vars[i] = (grad, var)

train_op = opt.apply_gradients(grad_vars)
# train_op = hvd.DistributedOptimizer(tf.train.AdagradOptimizer(lr * hvd.size())).minimize(loss)


config=tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = str(hvd.local_rank())

with tf.train.MonitoredTrainingSession(config=config, hooks=[hvd.BroadcastGlobalVariablesHook(0)]) as sess:
  # sess.run(tf.global_variables_initializer())
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
      last_time, last_k = time.time(), k
