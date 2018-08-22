#include <dirent.h>
#include <sys/stat.h>
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
    bool threadStop;

    void *cudaHostPtr;
    ssize_t cudaHostFloatCnt;

    Generator(const string &path, int height, int width, int cache_size, int thread_para): height(height), width(width), channel(3),
        cache_size(cache_size), tids(thread_para), cudaHostPtr(nullptr), cudaHostFloatCnt(0LU) {

      pthread_mutex_init(&m_lock, 0);
      threadStop = false;

      dirent *ep, *ch_ep;
      DIR *root = opendir(path.c_str());
      die_if(root == nullptr, "Cannot open directory of path: %s.", path.c_str());

      while ((ep = readdir(root)) != nullptr) {
        if (!ep->d_name[0] || !strcmp(ep->d_name, ".") || !strcmp(ep->d_name, ".."))
          continue;
        string sub_dir = path + ep->d_name + "/";
        DIR *child = opendir(sub_dir.c_str());
        if (child == nullptr)
          continue;
        while ((ch_ep = readdir(child)) != nullptr) {
          if (!ch_ep->d_name[0] || !strcmp(ch_ep->d_name, ".") || !strcmp(ch_ep->d_name, ".."))
            continue;
          dict[sub_dir].push_back(ch_ep->d_name);
        }
        closedir(child);
      }
      closedir(root);

      int samples = 0;
      for (auto &it: dict) {
        keyset.push_back(it.first);
        sort(keyset.begin(), keyset.end());
        samples += it.second.size();
      }
      n_class = keyset.size();

      printf("\nTotal %d samples found with %d classes for `file://%s`:\n", samples, n_class, path.c_str());
      die_if(!samples, "No valid samples found in directory.");
      for (int i = 0; i < n_class; ++i)
        printf("  (*) class %d => %s (%zd samples)\n", i, keyset[i].c_str(), dict[keyset[i]].size());

      for (int i = 0; i < tids.size(); ++i)
        pthread_create(&tids[i], NULL, Generator::start, this);
      __sync_add_and_fetch(&activeThread, tids.size());
    }

    ~Generator() {
      void *ret;
      threadStop = true;
      for (auto tid: tids)
        pthread_join(tid, &ret);
      pthread_mutex_destroy(&m_lock);
      if (cudaHostPtr != nullptr)
        assert(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
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
      return true;
    }

    void background_generator() {
      while (1) {
        vector<float> chw(channel * height * width), l(n_class, 0.0f);
        while (1) {
          int c = rand() % dict.size();
          auto &files = dict[keyset[c]];
          if (files.size() == 0)
            continue;
          int it = rand() % files.size();
          if (get_image_data(keyset[c] + files[it], chw.data(), c, l.data()))
            break;
        }

        while (1) {
          if (threadStop || globalStop) {
            __sync_add_and_fetch(&activeThread, -1);
            return;
          }
          pthread_mutex_lock(&m_lock);
          if (q_chw.size() >= cache_size) {
            pthread_mutex_unlock(&m_lock);
            usleep(50000);
          } else {
            q_chw.push(move(chw)), q_l.push(move(l));
            pthread_mutex_unlock(&m_lock);
            break;
          }
        }
      }
    }

    static void *start(void *arg) {
      ((Generator*)arg)->background_generator();
      return NULL;
    }

    auto next_batch(int batch_size = 32) {
      size_t split = batch_size * channel * height * width, tail = split + batch_size * keyset.size();

      if (cudaHostFloatCnt < tail) {
        if (cudaHostPtr != nullptr)
          assert(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
        cudaHostFloatCnt = tail;
        assert(CUDA_SUCCESS == cuMemHostAlloc(&cudaHostPtr, cudaHostFloatCnt * sizeof(float), 0));
      }
      // vector<float> nchw(batch_size * channel * height * width);
      // vector<float> nl(batch_size * keyset.size());

      int it = 0;
      while (it < batch_size) {
        pthread_mutex_lock(&m_lock);
        while (it < batch_size && q_chw.size()) {
          float *images = ((float*)cudaHostPtr) + (channel * height * width) * it;
          float *labels = ((float*)cudaHostPtr) + split + (n_class) * it;
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
        Tensor({batch_size, channel, height, width}, ((float*)cudaHostPtr)),
        Tensor({batch_size, n_class}, ((float*)cudaHostPtr) + split)
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
    void *cudaHostPtr;
    ssize_t cudaHostFloatCnt;

    Generator(): cudaHostPtr(nullptr), cudaHostFloatCnt(0) {
    }

    ~Generator() {
      if (cudaHostPtr != nullptr)
        assert(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
    }

    void save_to_directory(string path) {
      die_if(channel != 3 && channel != 1, "Not supporting image channel to save (channel = %d).", channel);
      if (path.back() != '/')
        path += '/';

      int it = path.find('/', 1);
      while (it >= 0) {
        mkdir(path.substr(0, it).c_str(), 0755);
        it = path.find('/', it + 1);
      }
      for (int i = 0; i < n_class; ++i)
        mkdir((path + to_string(i)).c_str(), 0755);

      int stride = (channel == 3) ? height * width : 0;
      for (int k = 0; k < n_sample; ++k) {
        cv::Mat dst(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
        uint8_t *ptr = dst.data;

        float *offset = images_data.data() + k * channel * height * width;
        int pred = 0;
        for (int i = 1; i < n_class; ++i)
          if (labels_data[k * n_class + i] > labels_data[k * n_class + pred])
            pred = i;
        float *r = offset, *g = r + stride, *b = g + stride;
        for (int i = 0; i < height; ++i) {
          for (int j = 0; j < width; ++j) {
            int id = i * width + j;
            int x = r[id] * 255.0f + 1e-8f;
            int y = g[id] * 255.0f + 1e-8f;
            int z = b[id] * 255.0f + 1e-8f;
            assert(x >= 0 && y >= 0 && z >= 0 && x <= 255 && y <= 255 && z <= 255);
            *ptr++ = x;
            *ptr++ = y;
            *ptr++ = z;
          }
        }
        cv::imwrite(path + to_string(pred) + "/" + to_string(k) + ".jpg", dst);
      }
    }

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

        // vector<float> nchw(batch_size * channel * height * width);
        // vector<float> nl(batch_size * n_class);
        ssize_t split = batch_size * channel * height * width, tail = split + batch_size * n_class;

        if (cudaHostFloatCnt < tail) {
          if (cudaHostPtr != nullptr)
            assert(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
          cudaHostFloatCnt = tail;
          assert(CUDA_SUCCESS == cuMemHostAlloc(&cudaHostPtr, cudaHostFloatCnt * sizeof(float), 0));
        }

        memcpy(((float*)cudaHostPtr), images_data.data() + index * channel * height * width, sizeof(float) * channel * height * width * (n_sample - index));
        memcpy(((float*)cudaHostPtr) + split, labels_data.data() + index * n_class, sizeof(float) * n_class * (n_sample - index));

        memcpy(((float*)cudaHostPtr) + channel * height * width * (n_sample - index), images_data.data(), sizeof(float) * channel * height * width * curr_iter);
        memcpy(((float*)cudaHostPtr) + split +  n_class * (n_sample - index), labels_data.data(), sizeof(float) * n_class * curr_iter);

        return dataset({
          Tensor({batch_size, channel, height, width}, ((float*)cudaHostPtr)),
          Tensor({batch_size, n_class}, ((float*)cudaHostPtr) + split)
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
      die_if(true, "Un supported dataset header format: %d\n", header);

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

    } else
      die_if(true, "Un supported dataset header format: %d\n", header);
    return {{}, {}};
  };

  auto full_images = ReadNormalDataset(images_ubyte);
  auto full_labels = ReadNormalDataset(labels_ubyte);
  die_if(full_images.first[0] != full_labels.first[0], "The number of images and labels in total doesn't match.");

  auto gen = make_unique<Generator>();

  gen->images_data = move(full_images.second);
  gen->labels_data = move(full_labels.second);

  gen->curr_iter = 0;
  gen->n_sample = full_labels.first[0];
  gen->n_class = full_labels.first[1], gen->channel = full_images.first[1], gen->height = full_images.first[2], gen->width = full_images.first[3];

  printf("Total %d samples found with %d classes.\n", gen->n_sample, gen->n_class);
  return move(gen);
}