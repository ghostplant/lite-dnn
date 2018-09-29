#include <dirent.h>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <queue>

class NormalGenerator {

public:
  struct Dataset {
    Tensor images, labels;
  };

  virtual Dataset next_batch(int batch_size) = 0;
  virtual vector<int> get_shape() = 0;
};

void make_dirs(const string &path) {
  die_if(path.back() != '/', "Directory path must end with '/'.");
  int it = path.find('/', 1);
  while (it >= 0) {
    mkdir(path.substr(0, it).c_str(), 0755);
    it = path.find('/', it + 1);
  }
}


auto synthetic_generator(int height, int width, int n_class, int channel = 3) {

  struct Generator: public NormalGenerator {
    int n_class, channel, height, width;

    Generator(int height, int width, int n_class, int channel): height(height), width(width), n_class(n_class), channel(channel) {
    }

    NormalGenerator::Dataset next_batch(int batch_size) {
      return NormalGenerator::Dataset({
        Tensor({batch_size, channel, height, width}),
        Tensor({batch_size, n_class})
      });
    }

    vector<int> get_shape() {
      return {n_class, channel, height, width};
    }
  };
  return make_unique<Generator>(height, width, n_class, channel);
}


auto iobuff_generator(const string &iobuffexec) {

  struct Generator: public NormalGenerator {
    int n_class, channel, height, width;
    FILE *fp;

    void *cudaHostPtr;
    ssize_t cudaHostFloatCnt;
    int magic;

    Generator(const string &iobuffexec): cudaHostPtr(nullptr), cudaHostFloatCnt(0LU), magic(0x7fbf00ff) {

      fp = popen(iobuffexec.c_str(), "r");
      ensure(fp != nullptr);

      int lchw[4];
      ensure(sizeof(lchw) == fread(lchw, 1, sizeof(lchw), fp));

      n_class = lchw[0];
      channel = lchw[1], height = lchw[2], width = lchw[3];

      printf("\nDetect Image Buffer from stdio with %d classes and image_shape = (%d, %d, %d).\n", lchw[0], lchw[1], lchw[2], lchw[3]);
    }

    ~Generator() {
      if (cudaHostPtr != nullptr)
        ensure(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
      if (fp != nullptr)
        pclose(fp);
    }

    NormalGenerator::Dataset next_batch(int batch_size) {
      size_t split = batch_size * channel * height * width, tail = split + batch_size * n_class;

      if (cudaHostFloatCnt < tail) {
        if (cudaHostPtr != nullptr)
          ensure(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
        cudaHostFloatCnt = tail;
        ensure(CUDA_SUCCESS == cuMemHostAlloc(&cudaHostPtr, cudaHostFloatCnt * sizeof(float), 0));
      }

      for (int offset = 0; offset < batch_size; ++offset) {
        float *images = ((float*)cudaHostPtr) + (channel * height * width) * offset;
        float *labels = ((float*)cudaHostPtr) + split + (n_class) * offset;
        int header = 0;
        ensure(1 == fread(&header, sizeof(header), 1, fp));
        ensure(header == magic);
        ensure(1 == fread(images, (channel * height * width) * sizeof(float), 1, fp));
        ensure(1 == fread(labels, (n_class) * sizeof(float), 1, fp));
        ++offset;
      }

      auto ds = NormalGenerator::Dataset({
        Tensor({batch_size, channel, height, width}),
        Tensor({batch_size, n_class})
      });
      ds.images.set_data(((float*)cudaHostPtr), false);
      ds.labels.set_data(((float*)cudaHostPtr) + split, true);
      return move(ds);
    }

    vector<int> get_shape() {
      return {n_class, channel, height, width};
    }
  };

  return make_unique<Generator>(iobuffexec);
}


auto image_generator(string path, int height = 229, int width = 229, int thread_para = 4) {
  die_if(thread_para > 32, "Too many thread workers for image_generator: %d.\n", thread_para);

  struct Generator: public NormalGenerator {
    unordered_map<string, vector<string>> dict;
    vector<string> keyset;
    int n_class, channel, height, width;

    struct Worker {
      pthread_t tid;
      pthread_mutex_t m_lock;
      queue<vector<float>> q_chw, q_l;
    };

    vector<Worker> workers;
    bool threadStop;
    int cache_size;

    void *cudaHostPtr;
    ssize_t cudaHostFloatCnt;

    Generator(const string &path, int height, int width, int thread_para): height(height), width(width), channel(3),
        workers(thread_para), cudaHostPtr(nullptr), cudaHostFloatCnt(0LU) {

      threadStop = false;
      cache_size = (1U << 30) / thread_para / (height * width * channel * sizeof(float));
      cache_size = max(1024, min(cache_size, (1 << 20)));

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

      __sync_add_and_fetch(&activeThread, workers.size());
      for (int i = 0; i < workers.size(); ++i) {
        pthread_mutex_init(&workers[i].m_lock, 0);
        pthread_create(&workers[i].tid, NULL, Generator::start, new pair<Generator*, int>(this, i));
      }
    }

    ~Generator() {
      threadStop = true;
      for (auto &worker: workers) {
        pthread_join(worker.tid, nullptr);
        pthread_mutex_destroy(&worker.m_lock);
      }
      if (cudaHostPtr != nullptr)
        ensure(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
    }


    void background_generator(int rank) {
      unsigned int seed = rank, mpi_rank = 0;
      const char *mpi_rank_env = getenv("OMPI_COMM_WORLD_RANK");
      if (mpi_rank_env)
        mpi_rank = atoi(mpi_rank_env);
      seed ^= mpi_rank << 16;
      // printf("Generating with seed: %u\n", seed);

      while (1) {
        vector<float> chw(channel * height * width), l(n_class, 0.0f);
        while (1) {
          int c = u_rand(&seed) % dict.size(); // rand_r(&seed)
          auto &files = dict[keyset[c]];
          if (files.size() == 0)
            continue;
          int it = u_rand(&seed) % files.size(); // rand_r(&seed)
          if (get_image_data(keyset[c] + files[it], chw.data(), c, l.data()))
            break;
        }

        while (1) {
          if (threadStop || globalStop) {
            // On Exit
            __sync_add_and_fetch(&activeThread, -1);
            return;
          }
          pthread_mutex_lock(&workers[rank].m_lock);
          if (workers[rank].q_chw.size() >= cache_size) {
            pthread_mutex_unlock(&workers[rank].m_lock);
            usleep(50000);
          } else {
            workers[rank].q_chw.push(move(chw));
            workers[rank].q_l.push(move(l));
            pthread_mutex_unlock(&workers[rank].m_lock);
            break;
          }
        }
      }
    }

    NormalGenerator::Dataset next_batch(int batch_size) {
      size_t split = batch_size * channel * height * width, tail = split + batch_size * keyset.size();

      if (cudaHostFloatCnt < tail) {
        if (cudaHostPtr != nullptr)
          ensure(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
        cudaHostFloatCnt = tail;
        ensure(CUDA_SUCCESS == cuMemHostAlloc(&cudaHostPtr, cudaHostFloatCnt * sizeof(float), 0));
      }

      ensure(cache_size * workers.size() >= batch_size);
      int para = workers.size();
      int batch_each = batch_size / para, batch_extra = batch_size % para;

      int offset = 0, rest = 0;
      for (int i = 0; i < para; ++i) {
        rest += batch_each + (i < batch_extra);
        while (offset < rest) {
          vector<vector<float>> f_chw, f_l;
          pthread_mutex_lock(&workers[i].m_lock);
          while (f_chw.size() < rest - offset && workers[i].q_chw.size()) {
            f_chw.push_back(move(workers[i].q_chw.front()));
            f_l.push_back(move(workers[i].q_l.front()));
            workers[i].q_chw.pop();
            workers[i].q_l.pop();
          }
          pthread_mutex_unlock(&workers[i].m_lock);

          while (f_chw.size()) {
            float *images = ((float*)cudaHostPtr) + (channel * height * width) * offset;
            float *labels = ((float*)cudaHostPtr) + split + (n_class) * offset;
            memcpy(images, f_chw.back().data(), (channel * height * width) * sizeof(float));
            memcpy(labels, f_l.back().data(), (n_class) * sizeof(float));
            f_chw.pop_back();
            f_l.pop_back();
            ++offset;
          }
        }
      }
      ensure(offset == batch_size);

      auto ds = NormalGenerator::Dataset({
        Tensor({batch_size, channel, height, width}),
        Tensor({batch_size, n_class})
      });
      ds.images.set_data(((float*)cudaHostPtr), false);
      ds.labels.set_data(((float*)cudaHostPtr) + split, true);
      return move(ds);
    }


    static void *start(void *args) {
      auto data = (pair<Generator*, int>*)args;
      Generator* object = data->first;
      int rank = data->second;
      delete data;
      object->background_generator(rank);
      return NULL;
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

    vector<int> get_shape() {
      return {n_class, channel, height, width};
    }
  };

  if (path.size() > 0 && path[path.size() - 1] != '/')
    path += '/';
  return make_unique<Generator>(path, height, width, thread_para);
}

auto array_generator(const char* images_ubyte, const char* labels_ubyte) {

  struct Generator: public NormalGenerator {
    vector<float> images_data, labels_data;
    int n_sample, n_class, channel, height, width;
    int curr_iter;
    void *cudaHostPtr;
    ssize_t cudaHostFloatCnt;

    Generator(): cudaHostPtr(nullptr), cudaHostFloatCnt(0) {
    }

    ~Generator() {
      if (cudaHostPtr != nullptr)
        ensure(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
    }

    void save_to_directory(string path) {
      die_if(channel != 3 && channel != 1, "Not supporting image channel to save (channel = %d).", channel);
      if (path.back() != '/')
        path += '/';

      make_dirs(path);
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
            ensure(x >= 0 && y >= 0 && z >= 0 && x <= 255 && y <= 255 && z <= 255);
            *ptr++ = x;
            *ptr++ = y;
            *ptr++ = z;
          }
        }
        cv::imwrite(path + to_string(pred) + "/" + to_string(k) + ".jpg", dst);
      }
    }

    vector<int> get_shape() {
      return {n_class, channel, height, width};
    }

    NormalGenerator::Dataset next_batch(int batch_size) {
      int index = curr_iter;
      if (curr_iter + batch_size <= n_sample) {
        curr_iter += batch_size;

        return NormalGenerator::Dataset({
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
            ensure(CUDA_SUCCESS == cuMemFreeHost(cudaHostPtr));
          cudaHostFloatCnt = tail;
          ensure(CUDA_SUCCESS == cuMemHostAlloc(&cudaHostPtr, cudaHostFloatCnt * sizeof(float), 0));
        }

        memcpy(((float*)cudaHostPtr), images_data.data() + index * channel * height * width, sizeof(float) * channel * height * width * (n_sample - index));
        memcpy(((float*)cudaHostPtr) + split, labels_data.data() + index * n_class, sizeof(float) * n_class * (n_sample - index));

        memcpy(((float*)cudaHostPtr) + channel * height * width * (n_sample - index), images_data.data(), sizeof(float) * channel * height * width * curr_iter);
        memcpy(((float*)cudaHostPtr) + split +  n_class * (n_sample - index), labels_data.data(), sizeof(float) * n_class * curr_iter);

        return NormalGenerator::Dataset({
          Tensor({batch_size, channel, height, width}, ((float*)cudaHostPtr)),
          Tensor({batch_size, n_class}, ((float*)cudaHostPtr) + split)
        });
      }
    }
  };
  
  auto ReadNormalDataset = [&](const char* dataset) -> pair<vector<int>, vector<float>> {
    auto read_uint32 = [&](FILE *fp) {
      uint32_t val;
      ensure(fread(&val, sizeof(val), 1, fp) == 1);
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
      ensure(fread(raw.data(), 1, raw.size(), fp) == raw.size());

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
        ensure(fread(raw.data(), 1, raw.size(), fp) == raw.size());
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
        ensure(fread(raw.data(), 1, raw.size(), fp) == raw.size());
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
