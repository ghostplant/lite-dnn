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

#include <queue>
#include <algorithm>

using std::queue;
using std::sort;

using namespace std;


class Model {
public:
  shared_ptr<Layer> top_layer;

  vector<Layer*> layers;
  unordered_map<const Layer*, int> index;

  Model(const shared_ptr<Layer> &top_layer): top_layer(top_layer), index({{top_layer.get(), 1}}) {
    queue<Layer*> que;
    que.push(top_layer.get());
    int indice = 1;

    while (que.size()) {
      auto layer = que.front(); que.pop();
      layers.push_back(layer);

      for (auto &parent: layer->parents) {
        if (!index.count(parent.get())) {
          index[parent.get()] = ++indice;
          que.push(parent.get());
        }
      }
    }
    for (auto &it: index)
      it.second = indice - it.second;

    sort(layers.begin(), layers.end(), [&](const Layer *x, const Layer *y) {
      if (x->depth != y->depth)
        return x->depth < y->depth;
      return index[x] < index[y];
    });

    size_t parameter_count = 0;
    putchar('\n');
    for (auto &layer: layers) {
      printf(" => layer-#%02d D(%d): %20s, output_shape: %15s, from:", index[layer], layer->depth, layer->to_string().c_str(),
        Tensor::stringify_shape(layer->get_output_shape(), 1).c_str());
      if (layer->parents.size() == 0)
        printf(" (none)");
      for (auto &parent: layer->parents)
        printf(" #%02d", index[parent.get()]);
      putchar('\n');
      for (auto &weight: layer->get_weights())
        parameter_count += weight.count();
    }
    printf("\nParameters: %zd\n\n", parameter_count);
  }

  bool load_weights_from_file(const char *weight_path) {
    FILE *fp = fopen(weight_path, "rb");
    bool success = (fp != nullptr);

    for (auto &layer: layers) {
      if (!success)
        break;
      for (auto &weight: layer->get_weights()) {
        vector<float> host(weight.count());
        if (host.size() != fread(host.data(), sizeof(float), host.size(), fp)) {
          success = false;
          break;
        }
        weight.set_data(host.data());
      }
    }

    if (success) {
      ssize_t offset = ftell(fp);
      fseek(fp, 0, SEEK_END);
      if (ftell(fp) != offset)
        success = false;
    }
    if (fp != nullptr)
      fclose(fp);
    printf("  [@] Loading weights data: %s.\n", success ? "YES" : "NO");
  }

  bool save_weights_to_file(const char *weight_path) {
    FILE *fp = fopen(weight_path, "wb");
    bool success = (fp != nullptr);

    for (auto &layer: layers) {
      if (!success)
        break;
      for (auto &weight: layer->get_weights()) {
        auto host = weight.get_data();
        if (host.size() != fwrite(host.data(), sizeof(float), host.size(), fp)) {
          success = false;
          break;
        }
      }
    }

    if (fp != nullptr)
      fclose(fp);
    printf("  [@] Saving weights data: %s.\n", success ? "YES" : "NO");
  }

  Tensor predict(const unordered_map<string, Tensor> &feed_dict = {}) {
    unordered_map<Layer*, Tensor> ys;

    for (auto &layer: layers) {
      vector<Tensor> xs;
      for (auto it: layer->parents) {
        assert(ys.count(it.get()) > 0);
        xs.push_back(ys[it.get()]);
      }
      assert(ys.count(layer) == 0);
      ys[layer] = layer->forward(xs, feed_dict);
    }
    return ys[top_layer.get()];
  }

  vector<Tensor> collect_all_gradients(const unordered_map<string, Tensor> &feed_dict) {
    unordered_map<Layer*, Tensor> dys = {{top_layer.get(), {}}};

    vector<Tensor> grads;
    for (int i = layers.size() - 1; i >= 0; --i) {
      auto &layer = layers[i];
      assert(dys.count(layer) > 0);

      auto curr = layer->get_gradients(dys[layer]);
      grads.insert(grads.end(), curr.begin(), curr.end());

      auto dxs = layer->backward(dys[layer], feed_dict);
      die_if(dxs.size() != layer->parents.size(), "the size of loss vector doesn't match the number of parent nodes.");

      for (int i = layer->parents.size() - 1; i >= 0; --i) {
        auto parent = layer->parents[i].get();

        if (dys.count(parent) == 0) {
          dys[parent] = dxs[i];
        } else {
          dys[parent].self_add(dxs[i]);
        }
      }
    }
    return move(grads);
  }

  vector<Tensor> collect_all_weights() {
    vector<Tensor> weights;
    for (int i = layers.size() - 1; i >= 0; --i) {
      auto &layer = layers[i];

      auto curr = layer->get_weights();
      if (curr.size() > 0)
        weights.insert(weights.end(), curr.begin(), curr.end());
    }
    return move(weights);
  }
};


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
  int batch_size = 64, steps = 10000;

  /*
  auto gen = array_generator(CIFAR10_IMAGES, CIFAR10_LABELS);

  // Mnist_MLP
  auto model = make_shared<InputLayer>("images_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>("labels_0"))
    ->summary();

  // Cifar10_Alexnet
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
  */

  die_if(0 != system("test -e /tmp/CatsAndDogs/.succ || (echo 'Downloading Cats-and-Dogs dataset ..' && curl -L https://github.com/ghostplant/public/releases/download/cats-and-dogs/cats-and-dogs.tar.gz | tar xzvf - -C /tmp >/dev/null && touch /tmp/CatsAndDogs/.succ)"),
      "Failed to download sample dataset.");

  // ImageNet_Resnet50v1
  /*
  auto gen = image_generator("/tmp/CatsAndDogs/train", 224, 224, 2048, 8);
  auto val_gen = image_generator("/tmp/CatsAndDogs/validate", 224, 224, 2048, 1);

  auto top_layer = make_shared<InputLayer>("images_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Convolution>(64, 7, 2, 3))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX));

  auto bottleneck_block_v1 = [&](shared_ptr<Layer> &input_layer, int depth, int depth_bottleneck, int stride) {
    auto shortcut = (depth == input_layer->get_output_shape()[1]) ? (stride == 1 ? input_layer : input_layer->then(make_shared<Pooling>(1, 2, CUDNN_POOLING_MAX)))
                      : input_layer->then(make_shared<Convolution>(depth, 1, stride))->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
    auto output = input_layer
      ->then(make_shared<Convolution>(depth_bottleneck, 1, stride))->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
      ->then(make_shared<Convolution>(depth_bottleneck, 3, 1, 1))->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
      ->then(make_shared<Convolution>(depth, 1, 1));

    return make_shared<Concat>(output, shortcut)
      ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
  };

  vector<int> layer_counts = {3, 4, 6, 3};
  for (int i = 0; i < layer_counts[0]; ++i)
    top_layer = bottleneck_block_v1(top_layer, 256, 64, 1);
  for (int i = 0; i < layer_counts[1]; ++i)
    top_layer = bottleneck_block_v1(top_layer, 512, 128, i == 0 ? 2 : 1);
  for (int i = 0; i < layer_counts[2]; ++i)
    top_layer = bottleneck_block_v1(top_layer, 1024, 256, i == 0 ? 2 : 1);
  for (int i = 0; i < layer_counts[3]; ++i)
    top_layer = bottleneck_block_v1(top_layer, 2048, 512, i == 0 ? 2 : 1);

  auto top_shape = top_layer->get_output_shape();
  die_if(top_shape.size() < 4 || top_shape[2] != top_shape[3], "Not supporting weight != height.");

  top_layer = top_layer->then(make_shared<Pooling>(top_shape[2], 1, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING))
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>("labels_0"));
  */

  // ImageNet_Alexnet
  auto gen = image_generator("/tmp/CatsAndDogs/train", 227, 227, 2048, 8);
  auto val_gen = image_generator("/tmp/CatsAndDogs/validate", 227, 227, 2048, 1);

  auto top_layer = make_shared<InputLayer>("images_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Convolution>(96, 11, 4))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
    ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
    ->then(make_shared<Convolution>(256, 5, 1, 2))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
    ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
    ->then(make_shared<Convolution>(384, 3, 1, 1))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
    ->then(make_shared<Convolution>(256, 3, 1, 1))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
    ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(4096))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dropout>(0.25))
    ->then(make_shared<Dense>(4096))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dropout>(0.25))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>("labels_0"));

  auto model = make_shared<Model>(top_layer);
  model->load_weights_from_file(weight_path);

  auto symbolic_weights = model->collect_all_weights();
  vector<Tensor> symbolic_velocity(symbolic_weights.size());
  for (int i = 0; i < symbolic_weights.size(); ++i)
    symbolic_velocity[i] = Tensor(symbolic_weights[i].shape, 0.0f);

  unsigned long lastClock = get_microseconds();
  for (int k = 0, it = 0; k < steps; ++k) {

    auto batch_data = gen->next_batch(batch_size);
    unordered_map<string, Tensor> feed_dict = {{"images_0", batch_data.images}, {"labels_0", batch_data.labels}};

    auto predicts = model->predict(feed_dict);

    auto symbolic_gradients = model->collect_all_gradients(feed_dict);
    die_if(symbolic_weights.size() != symbolic_gradients.size(), "The quantity of weight and gradient doesn't match.");

    // SGD
    // float lr = -float(0.05f * pow((1.0f + 0.0001f * k), -0.75f));
    // for (int i = 0; i < symbolic_weights.size(); ++i)
    //   symbolic_weights[i].self_add(symbolic_gradients[i], lr);

    // Momentum
    float momentum = 0.9f, lr = 0.01;
    for (int i = 0; i < symbolic_weights.size(); ++i) {
      symbolic_velocity[i].self_update(symbolic_gradients[i], lr, momentum);
      symbolic_weights[i].self_add(symbolic_velocity[i], -1.0f);
    }

    unsigned long currClock = get_microseconds();
    if (currClock >= lastClock + 1000000) {
      auto lacc = predicts.get_loss_and_accuracy_with(batch_data.labels);

      auto val_batch_data = val_gen->next_batch(batch_size);
      auto val_predicts = model->predict({{"images_0", val_batch_data.images}});
      auto val_lacc = val_predicts.get_loss_and_accuracy_with(val_batch_data.labels);

      printf("==> step = %d: loss = %.4f, acc = %.1f%%, val_loss = %.4f, val_acc = %.1f%%, time = %.2fs\n", k, lacc.first, lacc.second, val_lacc.first, val_lacc.second, (currClock - lastClock) * 1e-6f);
      lastClock = currClock;
    }
  }

  model->save_weights_to_file(weight_path);
  return 0;
}
