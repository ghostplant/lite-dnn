/*
  mnist_mlp based on CUBLAS/CUDNN
  g++ -O3 -std=c++14 "$@" -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcblas -lcudnn

  Maintainer: Wei CUI <ghostplant@qq.com>

  Benchmark on Nvida Tesla P100:

  ----------------------------------------------------------------------------
       Model        | batch_size  |    Keras + TF_CUDA    |  Lite-DNN (C++14)
  ----------------------------------------------------------------------------
     mnist_mlp      |    32       |    8.34 sec/epoll     |  1.03 sec/epoll
     mnist_cnn      |    128      |    3.24 sec/epoll     |  1.35 sec/epoll
     cifar10_lenet  |    128      |    2.68 sec/epoll     |  1.15 sec/epoll
  ----------------------------------------------------------------------------
*/


#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn_v7.h>

#include <core/tensor.h>
#include <core/layers.h>
#include <core/model.h>
#include <core/optimizor.h>
#include <core/multidev.h>

#include <core/generator.h>
#include <core/dataset.h>

#include <apps/resnet50.h>
#include <apps/alexnet.h>


static inline unsigned long get_microseconds() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000000LU + tv.tv_usec;
}


int main(int argc, char **argv) {
  Tensor::init();

  /* 
  auto gen = array_generator(CIFAR10_IMAGES, CIFAR10_LABELS);
  auto model = make_shared<InputLayer>("image_place_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>("label_place_0"))
    ->compile(); */

  int ngpus = 1, batch_size = 64, steps = 50000, sync_frequency = 8;
  DeviceEvents events;

  // auto ds = load_images("cifar10"); auto gen = image_generator(ds.first, 32, 32, 8), val_gen = image_generator(ds.second, 32, 32, 1);
  // auto gen = synthetic_generator(32, 32, 10);

  auto ds = load_images("catsdogs"); auto gen = image_generator(ds.first, 224, 224, 8), val_gen = image_generator(ds.second, 224, 224, 1);
  // auto gen = synthetic_generator(224, 224, 2);

  vector<shared_ptr<Model>> model_replias(ngpus);
  vector<shared_ptr<Optimizor>> optimizors(ngpus);

  for (int i = 0; i < ngpus; ++i) {
    Tensor::activateCurrentDevice(i);
    auto img_shape = gen->get_shape();
    model_replias[i] = lite_dnn::apps::imagenet_resnet50v1::
      create_model("image_place_0", "label_place_0", {img_shape[1], img_shape[2], img_shape[3]}, img_shape[0]);
    if (i == 0) {
      Tensor::activateCurrentDevice(0);
      model_replias[0]->load_weights_from_file("weights.lw");
    }

    optimizors[i] = make_shared<MomentumOptimizor>(model_replias[i]);
    // optimizors[i] = make_shared<SGDOptimizor>(model_replias[i], 0.005f);
  }

  unsigned long lastClock = get_microseconds();

  vector<vector<Tensor>> weights(ngpus);
  Tensor::activateCurrentDevice(0);
  weights[0] = model_replias[0]->collect_all_weights();
  for (int j = 1; j < ngpus; ++j) {
    weights[j] = model_replias[j]->collect_all_weights();
    for (int i = 0; i < weights[0].size(); ++i)
      weights[0][i].copyTo(weights[j][i]);
  }
  Tensor::synchronizeCurrentDevice();

  vector<Tensor> dst;

  for (int k = 1; k <= steps; ++k) {
    vector<Tensor> pred_label;

    for (int i = 0; i < ngpus; ++i) {
      Tensor::activateCurrentDevice(i);
      auto batch_data = gen->next_batch(batch_size);
      auto feed_dict = unordered_map<string, Tensor>({{"image_place_0", batch_data.images}, {"label_place_0", batch_data.labels}});

      auto predicts = model_replias[i]->predict(feed_dict);
      auto grad = model_replias[i]->collect_all_gradients(feed_dict);
      optimizors[i]->apply_updates(grad);

      if (i == 0)
        pred_label = { predicts, batch_data.labels };
    }

    // Strict sync every after `sync_frequency` times of batch training
    if (k % sync_frequency == 0 && ngpus > 1) {
      if (!dst.size()) {
        dst.resize(weights[0].size());
        for (int i = 0; i < dst.size(); ++i) {
          int managed = i % ngpus;
          Tensor::activateCurrentDevice(managed);
          dst[i] = Tensor(weights[0][i].shape);
        }
      }

      for (int c = 0; c < weights[0].size(); ++c) {
        int managed = c % ngpus;
        Tensor::activateCurrentDevice(managed);
        for (int i = 0; i < ngpus; ++i) {
          if (i == managed)
            continue;
          events.setDependency(managed, i);
          weights[i][c].copyTo(dst[c]);
          weights[managed][c].self_add(dst[c]);
        }
        weights[managed][c].self_mul(1.0f / ngpus);

        for (int i = 0; i < ngpus; ++i) {
          if (i == managed)
            continue;
          weights[managed][c].copyTo(weights[i][c]);
          events.setDependency(i, managed);
        }
      }
      events.recycle();
    }

    Tensor::activateCurrentDevice(0);

    unsigned long currClock = get_microseconds();
    // if (currClock >= lastClock + 1000000) {
    if (k % 100 == 0) {
      static double tot_seconds = 0.0;
      auto lacc = pred_label[0].get_loss_and_accuracy_with(pred_label[1]);

      auto val_batch_data = val_gen->next_batch(batch_size);
      auto val_predicts = model_replias[0]->predict({{"image_place_0", val_batch_data.images}});
      auto val_lacc = val_predicts.get_loss_and_accuracy_with(val_batch_data.labels);

      double seconds = (currClock - lastClock) * 1e-6f;
      tot_seconds += seconds;

      printf("==> step = %d (batch = %d; %.2f images/sec): loss = %.4f, acc = %.1f%%, val_loss = %.4f, val_acc = %.1f%%, time = %.2fs\n",
          k, batch_size * ngpus, k * batch_size / tot_seconds, lacc.first, lacc.second, val_lacc.first, val_lacc.second, seconds);
      lastClock = currClock;
    }
  }

  Tensor::activateCurrentDevice(0);
  model_replias[0]->save_weights_to_file("weights.lw");
  Tensor::quit();
  return 0;
}
