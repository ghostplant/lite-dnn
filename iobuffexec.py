#!/usr/bin/env python3

import numpy as np
import sys, warnings, time

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def create_dataset(data_dir, batch_size, height=224, width=224):
  # Prepare Dataset
  from keras.preprocessing.image import ImageDataGenerator
  from keras.utils import OrderedEnqueuer

  gen = ImageDataGenerator(data_format='channels_first', rescale=1./255, fill_mode='nearest').flow_from_directory(
                           data_dir, target_size=(height, width), batch_size=batch_size)

  enqueuer = OrderedEnqueuer(gen, use_multiprocessing=False)
  enqueuer.start(workers=16)
  return enqueuer, gen.num_classes, (3, height, width)


data_dir = sys.argv[1]
enqueuer, n_classes, image_shape = create_dataset(data_dir=data_dir, batch_size=1)

lchw = np.array((n_classes, ) + image_shape, dtype=np.int32).tobytes()

sys.stdout.buffer.write(lchw)

while True:
  batch_xs, batch_ys = next(enqueuer.get())
  x = batch_xs.tobytes()
  y = batch_ys.tobytes()
  try:
    sys.stdout.buffer.write(np.array([0x7fbf00ff], dtype=np.int32).tobytes())
    sys.stdout.buffer.write(x)
    sys.stdout.buffer.write(y)
  except:
    exit(0)
