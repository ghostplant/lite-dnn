#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import warnings, time

import keras
from keras import optimizers
from keras.applications import ResNet50
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import OrderedEnqueuer
from keras.utils import multi_gpu_model

def warn(*args, **kwargs):
    pass
warnings.warn = warn

height, width = 224, 224
num_classes = 2
batch_size = 64
steps = 50000


if K.image_data_format() == 'channels_first':
    image_shape = (3, height, width)
else:
    image_shape = (height, width, 3)

model = ResNet50(weights=None,
                 include_top=True,
                 input_shape=image_shape,
                 classes=num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01, momentum=0.9))

model.summary()


gen = ImageDataGenerator(
      data_format=K.image_data_format(),
      rescale=1./255,
      fill_mode='nearest').flow_from_directory('/tmp/dataset/catsdogs/train', target_size=(height, width), batch_size=batch_size)
enqueuer = OrderedEnqueuer(gen, use_multiprocessing=True)
enqueuer.start(workers=4)

init_time = last_time = time.time()
k = 1
for batch_xs, batch_ys in enqueuer.get():
  model.train_on_batch(batch_xs, batch_ys)
  curr_time = time.time()
  if k % 100 == 0:
  # if curr_time >= last_time + 1.0:
    out_loss = model.test_on_batch(batch_xs, batch_ys)
    print('step = %d (batch = %d; %.2f images/sec): loss = %.4f, time = %.3fs' % (k, batch_size, batch_size * (k + 1) / (curr_time - init_time), out_loss, curr_time - last_time))
    last_time = curr_time
  k = k + 1
  if k > steps:
    break
