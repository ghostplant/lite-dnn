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
      die_if(root == nullptr, "Cannot open directory of path: %s.", path.c_str());

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
        die_if(0 != pthread_create(&tids[i], NULL, Generator::start, this), "Failed to create intra-threads for data generation.");
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
