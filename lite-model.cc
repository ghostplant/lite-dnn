/*
  mnist_mlp based on CUBLAS/CUDNN

  Maintainer: Wei CUI <ghostplant@qq.com>

  Benchmark on Nvida Tesla P100:

  ---------------------------------------------------------------------------------
       Model            | batch_size  |    Keras + TF_CUDA    |  Lite-DNN (C++14)
  ---------------------------------------------------------------------------------
     mnist_mlp          |    32       |    8.34 sec/epoll     |  1.03 sec/epoll
     mnist_cnn          |    128      |    3.24 sec/epoll     |  1.35 sec/epoll
     cifar10_lenet      |    128      |    2.68 sec/epoll     |  1.15 sec/epoll
     imagenet_resnet50  |    64       |    149.10 images/sec  |  243.22 images/sec
  ---------------------------------------------------------------------------------
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

#include <nccl.h>

static inline unsigned long get_microseconds() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000000LU + tv.tv_usec;
}


int main(int argc, char **argv) {
  const int ngpus = Tensor::init();

  int devs[ngpus]; ncclComm_t comms[ngpus];
  for (int i = 0; i < ngpus; ++i)
    devs[i] = i;
  ensure(0 == ncclCommInitAll(comms, ngpus, devs));

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

  int batch_size = 64, steps = 50000;
  DeviceEvents events;

  // auto ds = load_images("cifar10"); auto gen = image_generator(ds.first, 32, 32, 8), val_gen = image_generator(ds.second, 32, 32, 1);
  // auto gen = synthetic_generator(32, 32, 10);

  auto ds = load_images("catsdogs"); auto gen = image_generator(ds.first, 224, 224, 8), val_gen = image_generator(ds.second, 224, 224, 1);
  // auto gen = synthetic_generator(224, 224, 2);

  // auto gen = image_generator("/var/lib/docker/imagenet/train", 224, 224, 8), val_gen = image_generator("/var/lib/docker/imagenet/validate", 224, 224, 1);

  vector<shared_ptr<Model>> model_replias(ngpus);
  vector<shared_ptr<Optimizor>> optimizors(ngpus);

  auto img_shape = gen->get_shape();
  for (int i = 0; i < ngpus; ++i) {
    Tensor::activateCurrentDevice(i);
    model_replias[i] = lite_dnn::apps::imagenet_resnet50v1::
      create_model("image_place_0", "label_place_0", {img_shape[1], img_shape[2], img_shape[3]}, img_shape[0]);

    if (i == 0) {
      Tensor::activateCurrentDevice(0);
      model_replias[0]->summary();
      model_replias[0]->load_weights_from_file("weights.lw");
    }

    optimizors[i] = make_shared<MomentumOptimizor>(model_replias[i]);
    // optimizors[i] = make_shared<SGDOptimizor>(model_replias[i], 0.005f);
  }

  unsigned long lastClock = get_microseconds();

  vector<vector<Tensor>> weights(ngpus), grad_reduce(ngpus), grad(ngpus);
  Tensor::activateCurrentDevice(0);
  weights[0] = model_replias[0]->collect_all_weights();
  for (int j = 1; j < ngpus; ++j) {
    weights[j] = model_replias[j]->collect_all_weights();
    for (int i = 0; i < weights[0].size(); ++i)
      weights[0][i].copyTo(weights[j][i]);
  }
  Tensor::synchronizeCurrentDevice();

  for (int i = 0; i < ngpus; ++i) {
    Tensor::activateCurrentDevice(i);
    grad_reduce[i].resize(weights[i].size());
    for (int j = 0; j < weights[i].size(); ++j)
      grad_reduce[i][j] = Tensor(weights[i][j].shape);
  }

  for (int k = 1; k <= steps; ++k) {
    vector<Tensor> pred_label;

    for (int i = 0; i < ngpus; ++i) {
      Tensor::activateCurrentDevice(i);
      auto batch_data = gen->next_batch(batch_size);
      auto feed_dict = unordered_map<string, Tensor>({{"image_place_0", batch_data.images}, {"label_place_0", batch_data.labels}});

      auto predicts = model_replias[i]->predict(feed_dict);
      grad[i] = model_replias[i]->collect_all_gradients(feed_dict);

      if (i == 0)
        pred_label = { predicts, batch_data.labels };
    }

    if (ngpus > 1) {
      // Gradient Reduce Sum
      ensure(0 == ncclGroupStart());
      for (int i = 0; i < ngpus; ++i) {
        Tensor::activateCurrentDevice(i);
        for (int j = 0; j < grad_reduce[i].size(); ++j) {
          ensure(0 == ncclAllReduce((const void*)grad[i][j].d_data->get(), (void*)grad_reduce[i][j].d_data->get(), grad_reduce[i][j].count(), ncclFloat, ncclSum, comms[i], devices[currentDev].hStream));
        }
      }
      ensure(0 == ncclGroupEnd());
    } else {
      for (int i = 0; i < ngpus; ++i)
        for (int j = 0; j < grad_reduce[i].size(); ++j)
          grad_reduce[i][j] = grad[i][j];
    }

    for (int i = 0; i < ngpus; ++i) {
      Tensor::activateCurrentDevice(i);
      optimizors[i]->apply_updates(grad_reduce[i]);
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
          k, batch_size * ngpus, k * ngpus * batch_size / tot_seconds, lacc.first, lacc.second, val_lacc.first, val_lacc.second, seconds);
      lastClock = currClock;
    }
  }

  Tensor::activateCurrentDevice(0);
  model_replias[0]->save_weights_to_file("weights.lw");

  for(int i = 0; i < ngpus; ++i)
    ensure(0 == ncclCommDestroy(comms[i]));
  Tensor::quit();
  return 0;
}
