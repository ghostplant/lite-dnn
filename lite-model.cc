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


#include <dirent.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <unordered_map>
#include <queue>

using std::unordered_map;

auto image_generator(string path, int height = 229, int width = 229, int cache_size = 256, int thread_para = 4) {

  struct Generator {
    unordered_map<string, vector<string>> dict;
    vector<string> keyset;
    int n_class, channel, height, width;
    queue<vector<float>> q_chw, q_l;

    int cache_size;
    vector<pthread_t> tids;
    pthread_mutex_t m_lock;
    bool thread_stop;

    Generator(const string &path, int height, int width, int cache_size, int thread_para): height(height), width(width), cache_size(cache_size), tids(thread_para), channel(3) {
      pthread_mutex_init(&m_lock, 0);
      thread_stop = 0;

      dirent *ep, *ch_ep;
      DIR *root = opendir(path.c_str());
      assert(root != nullptr);

      while ((ep = readdir(root)) != nullptr) {
        if (!ep->d_name[1] || (ep->d_name[1] == '.' && !ep->d_name[2]))
          continue;
        string sub_dir = path + ep->d_name + "/";
        DIR *child = opendir(sub_dir.c_str());
        if (child == nullptr)
          continue;
        while ((ch_ep = readdir(child)) != nullptr) {
          if (!ch_ep->d_name[1] || (ch_ep->d_name[1] == '.' && !ch_ep->d_name[2]))
            continue;
          dict[sub_dir].push_back(ch_ep->d_name);
        }
        closedir(child);
      }
      closedir(root);

      keyset.clear();
      int samples = 0;
      for (auto &it: dict) {
        keyset.push_back(it.first);
        samples += it.second.size();
      }
      n_class = keyset.size();

      printf("Total %d samples found with %d classes.\n", samples, n_class);

      for (int i = 0; i < tids.size(); ++i)
        assert(!pthread_create(&tids[i], NULL, Generator::start, this));
    }

    ~Generator() {
      void *ret;
      thread_stop = 1;
      for (auto tid: tids)
        pthread_join(tid, &ret);
      pthread_mutex_destroy(&m_lock);
    }


    bool get_image_data(const string &image_path, float *chw, int one_hot, float *l) {
      cv::Mat image = cv::imread(image_path, 1);
      if (image.data == nullptr)
        return false;

      cv::Size dst_size(height, width);
      cv::Mat dst;
      cv::resize(image, dst, dst_size);
      l[one_hot] = 1.0f;

      uint8_t *ptr = dst.data;
      float *b = chw, *g = b + height * width, *r = g + height * width;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          *b++ = *ptr++ / 255.0f, *g++ = *ptr++ / 255.0f, *r++ = *ptr++ / 255.0f;
        }
      }
      // cv::imwrite("/tmp/image-0.jpg", dst);
      return true;
    }

    void background_generator() {
      while (!thread_stop) {
        vector<float> chw(channel * height * width), l(n_class, 0.0f);
        while (1) {
          int c = rand() % dict.size();
          auto &files = dict[keyset[c]];
          int it = rand() % files.size();
          if (get_image_data(keyset[c] + files[it], chw.data(), c, l.data()))
            break;
        }

        while (!thread_stop) {
          pthread_mutex_lock(&m_lock);
          if (q_chw.size() >= cache_size) {
            pthread_mutex_unlock(&m_lock);
            if (thread_stop)
              return;
          } else
            break;
        }

        q_chw.push(move(chw)), q_l.push(move(l));
        pthread_mutex_unlock(&m_lock);
      }
    }

    static void *start(void *arg) {
      ((Generator*)arg)->background_generator();
      return NULL;
    }

    auto next_batch(int batch_size = 32) {
      vector<float> nchw(batch_size * channel * height * width);
      vector<float> nl(batch_size * keyset.size());

      int it = 0;
      while (it < batch_size) {
        pthread_mutex_lock(&m_lock);
        while (it < batch_size && q_chw.size()) {
          float *images = nchw.data() + (channel * height * width) * it;
          float *labels = nl.data() + (n_class) * it;
          memcpy(images, q_chw.front().data(), (channel * height * width) * sizeof(float));
          memcpy(labels, q_l.front().data(), (n_class) * sizeof(float));
          q_chw.pop();
          q_l.pop();
          ++it;
        }
        pthread_mutex_unlock(&m_lock);
      }

      struct dataset {
        Tensor images, labels;
      };

      return dataset({
        Tensor({batch_size, channel, height, width}, nchw.data()),
        Tensor({batch_size, n_class}, nl.data())
      });
    }
  };

  if (path.size() > 0 && path[path.size() - 1] != '/')
    path += '/';
  return make_unique<Generator>(path, height, width, cache_size, thread_para);
}

auto array_generator(const char* images_ubyte, const char* labels_ubyte) {

  struct Generator {
    vector<float> images_data, labels_data;
    int n_sample, n_class, channel, height, width;
    int curr_iter;

    auto next_batch(int batch_size = 32) {
      struct dataset {
        Tensor images, labels;
      };

      int index = curr_iter;
      if (curr_iter + batch_size <= n_sample) {
        curr_iter += batch_size;

        return dataset({
          Tensor({batch_size, channel, height, width}, images_data.data() + index * channel * height * width),
          Tensor({batch_size, n_class}, labels_data.data() + index * n_class)
        });
      } else {
        curr_iter += batch_size - n_sample;

        vector<float> nchw(batch_size * channel * height * width);
        vector<float> nl(batch_size * n_class);

        memcpy(nchw.data(), images_data.data() + index * channel * height * width, sizeof(float) * channel * height * width * (n_sample - index));
        memcpy(nl.data(), labels_data.data() + index * n_class, sizeof(float) * n_class * (n_sample - index));

        memcpy(nchw.data() + channel * height * width * (n_sample - index), images_data.data(), sizeof(float) * channel * height * width * curr_iter);
        memcpy(nl.data() +  n_class * (n_sample - index), labels_data.data(), sizeof(float) * n_class * curr_iter);

        return dataset({
          Tensor({batch_size, channel, height, width}, nchw.data()),
          Tensor({batch_size, n_class}, nl.data())
        });
      }
    }
  };
  
  auto ReadNormalDataset = [&](const char* dataset) -> pair<vector<int>, vector<float>> {
    auto read_uint32 = [&](FILE *fp) {
      uint32_t val;
      assert(fread(&val, sizeof(val), 1, fp) == 1);
      return __builtin_bswap32(val);
    };

    const int UBYTE_MAGIC = 0x800;
    FILE *fp;
    if ((fp = fopen(dataset, "rb")) == NULL) {
      printf("Cannot open file: %s\n", dataset);
      exit(1);
    }

    uint32_t header, length;
    header = read_uint32(fp);
    length = read_uint32(fp);
    header -= UBYTE_MAGIC;

    assert(header >= 1 && header <= 4);
    if (header == 1) { // output_shape = (N, max(val) + 1),  max(val) <= 255
      vector<uint8_t> raw(length);
      assert(fread(raw.data(), 1, raw.size(), fp) == raw.size());

      uint32_t width = 0;
      for (int i = 0; i < raw.size(); ++i)
        width = max(width, (uint32_t)raw[i]);
      ++width;

      vector<int> shape = {(int)length, (int)width};
      vector<float> tensor(length * width);
      for (int i = 0; i < length; ++i)
        tensor[i * width + raw[i]] = 1.0f;
      return {move(shape), move(tensor)};

    } else if (header == 2) { // shape = (N, C),  may support max(val) > 255
      assert(0); // unsupported

    } else if (header == 3) { // shape = (N, 1, H, W)
      uint32_t h = read_uint32(fp);
      uint32_t w = read_uint32(fp);
      uint32_t width = h * w;

      vector<int> shape = {(int)length, 1, (int)h, (int)w};
      vector<float> tensor(length * width);
      vector<uint8_t> raw(width);
      for (int i = 0; i < length; ++i) {
        assert(fread(raw.data(), 1, raw.size(), fp) == raw.size());
        for (int j = 0; j < width; ++j)
          tensor[i * width + j] = raw[j] / 255.0f;
      }
      return {move(shape), move(tensor)};

    } else if (header == 4) { // shape = (N, C, H, W)
      uint32_t c = read_uint32(fp);
      uint32_t h = read_uint32(fp);
      uint32_t w = read_uint32(fp);
      uint32_t width = c * h * w;

      vector<int> shape = {(int)length, (int)c, (int)h, (int)w};
      vector<float> tensor(length * width);
      vector<uint8_t> raw(width);
      for (int i = 0; i < length; ++i) {
        assert(fread(raw.data(), 1, raw.size(), fp) == raw.size());
        for (int j = 0; j < width; ++j)
          tensor[i * width + j] = raw[j] / 255.0f;
      }
      return {move(shape), move(tensor)};

    }
    assert(0);
    return {{}, {}};
  };

  auto full_images = ReadNormalDataset(images_ubyte);
  auto full_labels = ReadNormalDataset(labels_ubyte);
  assert(full_images.first[0] == full_labels.first[0]);

  auto gen = make_unique<Generator>();

  gen->images_data = move(full_images.second);
  gen->labels_data = move(full_labels.second);

  gen->curr_iter = 0;
  gen->n_sample = full_labels.first[0];
  gen->n_class = full_labels.first[1], gen->channel = full_images.first[1], gen->height = full_images.first[2], gen->width = full_images.first[3];

  printf("Total %d samples found with %d classes.\n", gen->n_sample, gen->n_class);
  return move(gen);
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
