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

#include <core/tensor.h>
#include <core/layers.h>
#include <core/model.h>
#include <core/optimizor.h>

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

  int batch_size = 64, steps = 50000;
  Tensor::activateCurrentDevice(0);

  // * Mnist_MLP
  /*auto gen = array_generator(MNIST_IMAGES, MNIST_LABELS), &val_gen = gen;

  auto model = make_shared<InputLayer>("image_place_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>("label_place_0"))
    ->compile();*/

  // * ImageNet_AlexNet
  die_if(0 != system("test -e /tmp/CatsAndDogs/.succ || (echo 'Downloading Cats-and-Dogs dataset ..' && curl -L https://github.com/ghostplant/public/releases/download/cats-and-dogs/cats-and-dogs.tar.gz | tar xzvf - -C /tmp >/dev/null && touch /tmp/CatsAndDogs/.succ)"), "Failed to download sample dataset.");
  auto gen = image_generator("/tmp/CatsAndDogs/train", 224, 224, 2048, 8),
         val_gen = image_generator("/tmp/CatsAndDogs/validate", 224, 224, 2048, 1);


  auto model = lite_dnn::apps::imagenet_alexnet::
         create_model("image_place_0", "label_place_0", {gen->channel, gen->height, gen->width}, gen->n_class);

  model->load_weights_from_file("weights.lw");

  auto optimizor = MomentumOptimizor(0.9f, 0.01f);
  auto symbolic_weights = model->collect_all_weights();

  unsigned long lastClock = get_microseconds();
  for (int k = 0; k < steps; ++k) {

    auto batch_data = gen->next_batch(batch_size);
    unordered_map<string, Tensor> feed_dict = {{"image_place_0", batch_data.images}, {"label_place_0", batch_data.labels}};

    auto predicts = model->predict(feed_dict);
    auto symbolic_gradients = model->collect_all_gradients(feed_dict);
    optimizor.apply_updates(symbolic_weights, symbolic_gradients);

    unsigned long currClock = get_microseconds();
    if (currClock >= lastClock + 1000000) {
      auto lacc = predicts.get_loss_and_accuracy_with(batch_data.labels);

      auto val_batch_data = val_gen->next_batch(batch_size);
      auto val_predicts = model->predict({{"image_place_0", val_batch_data.images}});
      auto val_lacc = val_predicts.get_loss_and_accuracy_with(val_batch_data.labels);

      printf("==> step = %d: loss = %.4f, acc = %.1f%%, val_loss = %.4f, val_acc = %.1f%%, time = %.2fs\n", k, lacc.first, lacc.second, val_lacc.first, val_lacc.second, (currClock - lastClock) * 1e-6f);
      lastClock = currClock;
    }
  }

  model->save_weights_to_file("weights.lw");
  return 0;
}
