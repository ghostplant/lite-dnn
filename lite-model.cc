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

/*
vector<shared_ptr<Layer>> create_model(const char *model, int n_class) {
  vector<shared_ptr<Layer>> layers;
  if (!strcmp(model, "mnist_mlp")) {
    layers.push_back(make_shared<InputLayer>(1, 28, 28));
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(512));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(512));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(n_class));
    layers.push_back(make_shared<SoftmaxCrossEntropy>());
  } else if (!strcmp(model, "mnist_cnn")) {
    layers.push_back(make_shared<InputLayer>(1, 28, 28));
    layers.push_back(make_shared<Convolution>(32, 3));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Convolution>(64, 3));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    layers.push_back(make_shared<Dropout>(0.25));
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(128));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dropout>(0.25));
    layers.push_back(make_shared<Dense>(n_class));
    layers.push_back(make_shared<SoftmaxCrossEntropy>());
  } else if (!strcmp(model, "cifar10_lenet")) {
    layers.push_back(make_shared<InputLayer>(3, 32, 32));
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
    layers.push_back(make_shared<SoftmaxCrossEntropy>());
  } else if (!strcmp(model, "cifar10_alexnet")) {
    layers.push_back(make_shared<InputLayer>(3, 32, 32));
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
    layers.push_back(make_shared<SoftmaxCrossEntropy>());
  } else if (!strcmp(model, "imagenet_vgg16")) {
    layers.push_back(make_shared<InputLayer>(3, 224, 224));
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
    layers.push_back(make_shared<Convolution>(256, 3, 1, 1));
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
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    // Include top
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(4096));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(4096));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(n_class));
    layers.push_back(make_shared<SoftmaxCrossEntropy>());
  } else {
    printf("No model of name %s found.\n", model);
    exit(1);
  }
  return move(layers);
}
*/

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

  const char *weight_path = "weights.lw";
  int batch_size = 128, steps = 60000;
  // auto gen = image_generator("/docker/PetImages/Pics", 224, 224, 1024, 8);
  auto gen = array_generator(CIFAR10_IMAGES, CIFAR10_LABELS);

  /*auto model = make_shared<InputLayer>(gen->channel, gen->height, gen->width)
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>());*/

  auto model = make_shared<InputLayer>("images_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Convolution>(64, 5, true))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
    ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
    ->then(make_shared<Convolution>(64, 5, true))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
    ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(384))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(192))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>())
    ->summary();

  model->load_weights_from_file(weight_path);

  unsigned long lastClock = get_microseconds();
  for (int k = 0, it = 0; k < steps; ++k) {

    float lr = -float(0.05f * pow((1.0f + 0.0001f * k), -0.75f));
    auto batch = gen->next_batch(batch_size);

    auto predicts = model->predict_on_batch({{"images_0", batch.images}});

    auto symbolic_weights = model->compute_all_weights();
    auto symbolic_gradients = model->compute_all_gradients(batch.labels);
    die_if(symbolic_weights.size() != symbolic_gradients.size(), "The quantities of weight and gradient don't match.");

    for (int i = 0; i < symbolic_weights.size(); ++i)
      symbolic_weights[i].self_add(symbolic_gradients[i], lr);

    unsigned long currClock = get_microseconds();
    if (currClock >= lastClock + 1000000) {
      auto loss_acc = get_loss_and_accuracy(predicts, batch.labels);
      printf("==> step = %d: lr = %.4f, loss = %.4f, accuracy = %.2f%%, time = %.4fs\n", k, lr, loss_acc.first, loss_acc.second, (currClock - lastClock) * 1e-6f);
      lastClock = currClock;
    }
  }

  model->save_weights_to_file(weight_path);
  return 0;
}
