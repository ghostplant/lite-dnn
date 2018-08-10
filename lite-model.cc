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

  // MLP
  /*auto model = make_shared<InputLayer>("images_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>("labels_0"))
    ->summary();*/

  // Alexnet
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
    ->then(make_shared<SoftmaxCrossEntropy>("labels_0"))
    ->summary();

  model->load_weights_from_file(weight_path);

  unsigned long lastClock = get_microseconds();
  for (int k = 0, it = 0; k < steps; ++k) {

    auto batch_data = gen->next_batch(batch_size);
    unordered_map<string, Tensor> feed_dict = {{"images_0", batch_data.images}, {"labels_0", batch_data.labels}};

    auto predicts = model->predict(feed_dict);

    auto symbolic_weights = model->collect_all_weights();
    auto symbolic_gradients = model->collect_all_gradients(feed_dict);
    die_if(symbolic_weights.size() != symbolic_gradients.size(), "The quantity of weight and gradient doesn't match.");

    float lr = -float(0.05f * pow((1.0f + 0.0001f * k), -0.75f));
    for (int i = 0; i < symbolic_weights.size(); ++i)
      symbolic_weights[i].self_add(symbolic_gradients[i], lr);

    unsigned long currClock = get_microseconds();
    if (currClock >= lastClock + 1000000) {
      auto loss_acc = predicts.get_loss_and_accuracy_with(batch_data.labels);
      printf("==> step = %d: lr = %.4f, loss = %.4f, accuracy = %.2f%%, time = %.2fs\n", k, lr, loss_acc.first, loss_acc.second, (currClock - lastClock) * 1e-6f);
      lastClock = currClock;
    }
  }

  model->save_weights_to_file(weight_path);
  return 0;
}
