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

#include <core/generator.h>
#include <core/dataset.h>

#include <apps/resnet50.h>
#include <apps/alexnet.h>


static inline unsigned long get_microseconds() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000000LU + tv.tv_usec;
}


#define MNIST_IMAGES "/tmp/mnist-images-idx3-ubyte"
#define MNIST_LABELS "/tmp/mnist-labels-idx1-ubyte"

#define CIFAR10_IMAGES "/tmp/cifar10-images-idx4-ubyte"
#define CIFAR10_LABELS "/tmp/cifar10-labels-idx1-ubyte"

int main(int argc, char **argv) {
  Tensor::init();

  int ngpus = 1;
  int batch_size = 64, steps = 50000;

  auto train_val = load_images("cifar10");
  // opt: cifar10 = 350 * 64 * 4 images/ sec; file: cifar10 = 300 * 64 * 4 images/ sec
  // auto gen = array_generator(CIFAR10_IMAGES, CIFAR10_LABELS);
  auto gen = image_generator(train_val.first, 32, 32, 8);

  /* auto model = make_shared<InputLayer>("image_place_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>("label_place_0"))
    ->compile(); */

  vector<shared_ptr<Model>> model_replias(ngpus);
  vector<shared_ptr<Optimizor>> optimizors(ngpus);

  for (int i = 0; i < ngpus; ++i) {
    Tensor::activateCurrentDevice(i);
    auto img_shape = gen->get_shape();
    model_replias[i] = lite_dnn::apps::cifar10_alexnet::
      create_model("image_place_0", "label_place_0", {img_shape[1], img_shape[2], img_shape[3]}, img_shape[0]);
    if (i == 0) {
      Tensor::activateCurrentDevice(0);
      model_replias[0]->load_weights_from_file("weights.lw");
    }

    optimizors[i] = make_shared<SGDOptimizor>(model_replias[i], 0.01f, 0.001f);
  }

  unsigned long lastClock = get_microseconds();

  vector<vector<Tensor>> grads(ngpus);
  Tensor::activateCurrentDevice(0);
  auto ws = model_replias[0]->collect_all_weights();
  for (int j = 1; j < ngpus; ++j) {
    auto wj = model_replias[j]->collect_all_weights();
    for (int i = 0; i < ws.size(); ++i)
      ws[i].copyTo(wj[i]);
  }
  Tensor::synchronizeCurrentDevice();

  for (int k = 0; k < steps; ++k) {
    for (int i = 0; i < ngpus; ++i) {
      Tensor::activateCurrentDevice(i);
      auto batch_data = gen->next_batch(batch_size);
      auto feed_dict = unordered_map<string, Tensor>({{"image_place_0", batch_data.images}, {"label_place_0", batch_data.labels}});

      auto predicts = model_replias[i]->predict(feed_dict);
      grads[i] = model_replias[i]->collect_all_gradients(feed_dict);
    }

    /* vector<vector<float>> parameters(grads[0].size());
    for (int j = 0; j < parameters.size(); ++j)
      parameters[j].resize(grads[0][j].count());
    for (int i = 0; i < ngpus; ++i) {
      Tensor::activateCurrentDevice(i);
      for (int j = 0; j < parameters.size(); ++j) {
        auto param = grads[i][j].get_data();
        ensure(param.size() == parameters[j].size());
        for (int k = 0; k < param.size(); ++k) {
          parameters[j][k] = parameters[j][k] * i / (i + 1.0f) + param[k] * 1.0f / (i + 1.0f);
        }
      }
    }
    for (int i = 0; i < ngpus; ++i) {
      Tensor::activateCurrentDevice(i);
      for (int j = 0; j < parameters.size(); ++j)
        grads[i][j].set_data(parameters[j].data());
    } */
    for (int i = 0; i < ngpus; ++i) {
      Tensor::activateCurrentDevice(i);
      optimizors[i]->apply_updates(grads[i]);
    }

    Tensor::activateCurrentDevice(0);

    unsigned long currClock = get_microseconds();
    if (currClock >= lastClock + 1000000) {
      int dev = 0;
      Tensor::activateCurrentDevice(dev);
      auto val_batch_data = gen->next_batch(batch_size);
      auto val_predicts = model_replias[dev]->predict({{"image_place_0", val_batch_data.images}});
      auto val_lacc = val_predicts.get_loss_and_accuracy_with(val_batch_data.labels);

      printf("==> step = %d: val_loss = %.4f, val_acc = %.1f%%, time = %.2fs\n", k, val_lacc.first, val_lacc.second, (currClock - lastClock) * 1e-6f);
      lastClock = currClock;
    }
  }

  Tensor::activateCurrentDevice(0);
  model_replias[0]->save_weights_to_file("weights.lw");
  Tensor::quit();
  return 0;
}
