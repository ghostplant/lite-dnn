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


#define MNIST_IMAGES "/tmp/mnist-images-idx3-ubyte"
#define MNIST_LABELS "/tmp/mnist-labels-idx1-ubyte"

#define CIFAR10_IMAGES "/tmp/cifar10-images-idx4-ubyte"
#define CIFAR10_LABELS "/tmp/cifar10-labels-idx1-ubyte"

#define TRAIN_IMAGES MNIST_IMAGES
#define TRAIN_LABELS MNIST_LABELS


vector<shared_ptr<Layer>> create_model(const char *model) {
  vector<shared_ptr<Layer>> layers;
  if (!strcmp(model, "mnist_mlp")) {
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(512));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dropout>(0.1));
    layers.push_back(make_shared<Dense>(512));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dropout>(0.1));
    layers.push_back(make_shared<Dense>(10));
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
    layers.push_back(make_shared<Dense>(10));
    layers.push_back(make_shared<Softmax>());
  } else if (!strcmp(model, "cifar10_lenet")) {
    layers.push_back(make_shared<Convolution>(32, 5, true));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    layers.push_back(make_shared<Convolution>(64, 5, true));
    layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
    layers.push_back(make_shared<Flatten>());
    layers.push_back(make_shared<Dense>(512));
    layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    layers.push_back(make_shared<Dense>(10));
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
    layers.push_back(make_shared<Dense>(10));
    layers.push_back(make_shared<Softmax>());
  } else {
    fprintf(stderr, "No model of name %s found.\n", model);
    exit(1);
  }
  return move(layers);
}

static inline unsigned long get_microseconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000LU + tv.tv_usec;
}

static unsigned long lastClock = get_microseconds();


int main(int argc, char **argv) {
  Tensor<float>::init();

  auto full_images = ReadNormalDataset(TRAIN_IMAGES);
  auto full_labels = ReadNormalDataset(TRAIN_LABELS);
  assert(full_images.first[0] == full_labels.first[0]);

  int samples = full_images.first[0];
  int width = full_images.first[1] * full_images.first[2] * full_images.first[3];
  int classes = full_labels.first[1];

  const char *name = argc > 1 ? argv[1] : "mnist_cnn";
  printf("Total %d samples (%d, %d, %d) for %d classes found, using %s.\n", samples, full_images.first[1], full_images.first[2], full_images.first[3], classes, name);
  auto model = create_model(name);

  vector<Tensor<float>> input(model.size() + 1), dloss(model.size());
  int batch_size = 128, epochs = 50, steps = (samples + batch_size - 1) / batch_size * epochs;

  vector<int> shape = full_images.first;
  shape[0] = batch_size;
  puts("");
  for (int i = 0; i < model.size(); ++i) {
    shape = model[i]->configure(shape);
    printf("%12s, shape = (", model[i]->to_string().c_str());
    for (int i = 0; i < shape.size(); ++i)
      printf("%d, ", shape[i]);
    puts(")");
  }
  puts("");

  for (int k = 0, it = 0; k < steps; ++k) {
    vector<float> in(width * batch_size), out(classes * batch_size);
    for (int i = 0; i < batch_size; ++i, it = (it + 1) % samples) {
      assert(i * width + width <= in.size() && it * width + width <= full_images.second.size());
      assert(i * classes + classes <= in.size() && it * classes + classes <= full_images.second.size());
      memcpy(&in[i * width], &full_images.second[it * width], width * sizeof(float));
      memcpy(&out[i * classes], &full_labels.second[it * classes], classes * sizeof(float));
    }
    Tensor<float> images({batch_size, full_images.first[1], full_images.first[2], full_images.first[3]}, in), labels({batch_size, classes}, out);

    float lr = - float(0.05f * pow((1.0f + 0.0001f * k), -0.75f));

    input[0] = images;
    for (int i = 0; i < model.size(); ++i)
      input[i + 1] = model[i]->forward(input[i]);
    auto data_output = input.back();

    dloss[model.size() - 1] = model.back()->backward(input.back(), labels, input.back());
    for (int i = model.size() - 2; i >= 0; --i)
      dloss[i] = model[i]->backward(dloss[i + 1], input[i + 1], input[i], i == 0), model[i]->learn(lr);
    auto data_loss = dloss.back();

    if (it < batch_size) {
      int tot = 0, acc = 0;
      vector<float> pred_data = data_output.get_data();
      for (int i = 0; i < batch_size; ++i) {
        int it = 0, jt = 0;
        for (int j = 1; j < classes; ++j) {
          if (pred_data[i * classes + it] < pred_data[i * classes + j])
            it = j;
          if (out[i * classes + jt] < out[i * classes + j])
            jt = j;
        }
        ++tot;
        if (it == jt)
          ++acc;
      }

      vector<float> loss_data = data_loss.get_data();
      float loss = 0.0f;
      for (int i = 0; i < loss_data.size(); ++i) {
        float j = fabs(loss_data[i]);
        if (j >= 1e-8)
          loss += -j * log(j);
      }
      loss /= data_loss.shape[0];

      static int epoch = 0;
      unsigned long currClock = get_microseconds();
      printf("epoch = %d: loss = %.4f, acc = %.2f%%, time = %.4fs\n", ++epoch, loss, acc * 100.0f / tot, (currClock - lastClock) * 1e-6f);
      lastClock = currClock;
    }
  }
  return 0;
}
