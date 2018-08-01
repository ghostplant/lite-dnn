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
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn_v7.h>

#include <tensor.h>
#include <layers.h>
#include <dataset.h>

using namespace std;


vector<shared_ptr<Layer>> create_model(const char *model, int n_class) {
  vector<shared_ptr<Layer>> layers;
  if (!strcmp(model, "mnist_mlp")) {
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(512));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    // layers.push_back(make_shared<Dropout>(0.1));
    layers.push_back(make_shared<Dense>(512));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    // layers.push_back(make_shared<Dropout>(0.1));
    layers.push_back(make_shared<Dense>(n_class));
    layers.push_back(make_shared<Softmax>());
  } else if (!strcmp(model, "mnist_cnn")) {
    layers.push_back(make_shared<Convolution>(32, 3));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(64, 3));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    layers.push_back(make_shared<Dropout>(0.25));
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(128));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dropout>(0.5));
    layers.push_back(make_shared<Dense>(n_class));
    layers.push_back(make_shared<Softmax>());
  } else if (!strcmp(model, "cifar10_lenet")) {
    layers.push_back(make_shared<Convolution>(32, 5, true));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    layers.push_back(make_shared<Convolution>(64, 5, true));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    layers.push_back(make_shared<Dropout>(0.25));
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(512));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dropout>(0.25));
    layers.push_back(make_shared<Dense>(n_class));
    layers.push_back(make_shared<Softmax>());
  } else if (!strcmp(model, "cifar10_alexnet")) {
    layers.push_back(make_shared<Convolution>(64, 5, true));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX));
    layers.push_back(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75));
    layers.push_back(make_shared<Convolution>(64, 5, true));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75));
    layers.push_back(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX));
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(384));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(192));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(n_class));
    layers.push_back(make_shared<Softmax>());
  } else if (!strcmp(model, "cifar10_vgg16")) {
    // Block-1
    layers.push_back(make_shared<Convolution>(64, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(64, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    // Block-2
    layers.push_back(make_shared<Convolution>(128, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(128, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    // Block-3
    /*layers.push_back(make_shared<Convolution>(256, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(256, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(256, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    // Block-4
    layers.push_back(make_shared<Convolution>(512, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(512, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(512, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    // Block-5
    layers.push_back(make_shared<Convolution>(512, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(512, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(512, 3, 1, 1));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));*/
    // Include top
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(4096));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(4096));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(n_class));
    layers.push_back(make_shared<Softmax>());
  } else {
    printf("No model of name %s found.\n", model);
    exit(1);
  }
  return move(layers);
}

static inline unsigned long get_microseconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000LU + tv.tv_usec;
}


#define MNIST_IMAGES "/tmp/mnist-images-idx3-ubyte"
#define MNIST_LABELS "/tmp/mnist-labels-idx1-ubyte"

#define CIFAR10_IMAGES "/tmp/cifar10-images-idx4-ubyte"
#define CIFAR10_LABELS "/tmp/cifar10-labels-idx1-ubyte"

int main(int argc, char **argv) {
  Tensor::init();

  // auto gen = image_generator("/docker/PetImages/Pics", 32, 32, 1024, 8);
  auto gen = array_generator(CIFAR10_IMAGES, CIFAR10_LABELS);

  int batch_size = 128, steps = 60000;
  vector<int> shape = {batch_size, gen->channel, gen->height, gen->width};

  auto model = create_model(argc > 1 ? argv[1] : "mnist_cnn", gen->n_class);

  vector<Tensor> input(model.size() + 1), dloss(model.size());
  static unsigned long lastClock = get_microseconds();

  for (int k = 0, it = 0; k < steps; ++k) {
    auto batch = gen->next_batch(batch_size); auto &images = batch.images, &labels = batch.labels;

    float lr = - float(0.05f * pow((1.0f + 0.0001f * k), -0.75f));

    input[0] = images;
    for (int i = 0; i < model.size(); ++i)
      input[i + 1] = model[i]->forward(input[i]);
    auto data_output = input.back();

    dloss[model.size() - 1] = model.back()->backward(input.back(), labels, input.back());
    for (int i = model.size() - 2; i >= 0; --i)
      dloss[i] = model[i]->backward(dloss[i + 1], input[i + 1], input[i], i == 0), model[i]->learn(lr);
    auto data_loss = dloss.back();

    unsigned long currClock = get_microseconds();
    if (currClock >= lastClock + 1000000) {
      auto loss_acc = get_loss_and_accuracy(data_output, labels);
      printf("step = %d: lr = %.4f, loss = %.4f, accuracy = %.2f%%, time = %.4fs\n", k, lr, loss_acc.first, loss_acc.second, (currClock - lastClock) * 1e-6f);
      lastClock = currClock;
    }
  }
  return 0;
}
