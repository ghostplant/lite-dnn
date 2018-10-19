#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include <unordered_map>

#include <cuda.h>
#include <cuda_runtime.h>

#define die_if(__cond__, __desc__, ...) ({if (__cond__) { printf("  \033[33m[!] <<file %s:%d>> " __desc__ "\033[0m\n\n", __FILE__, __LINE__, ##__VA_ARGS__); fflush(stdout); exit(1);}})
#define ensure(__cond__)  die_if(!(__cond__), "Condition checking failed: %s.", #__cond__)

using namespace std;


static int activeThread = 0, hvd_rank = -1;

class MemoryManager {

  void* (*mem_alloc)(size_t);
  void (*mem_free)(void*);

  unordered_map<unsigned int, vector<void*>> resources;
  unordered_map<void*, unsigned int> buffsize;
  pthread_mutex_t m_lock;

public:
  MemoryManager(void* (*mem_alloc)(size_t), void (*mem_free)(void*)) {
    this->mem_alloc = mem_alloc;
    this->mem_free = mem_free;
    pthread_mutex_init(&m_lock, 0);
  }

  ~MemoryManager() {
    clear();
    pthread_mutex_destroy(&m_lock);
  }

  void clear() {
    for (auto &it: resources)
      for (auto &ptr: it.second)
        mem_free(ptr);
    resources.clear();
    buffsize.clear();
  }

  void* allocate(unsigned int floatCnt) {
    pthread_mutex_lock(&m_lock);
    ensure(floatCnt > 0);
    void *memptr;
    auto it = resources.find(floatCnt);
    if (it != resources.end() && it->second.size() > 0) {
      memptr = it->second.back();
      it->second.pop_back();
    } else {
      memptr = mem_alloc(size_t(floatCnt) * sizeof(float));
      ensure(buffsize.count(memptr) == 0);
      buffsize[memptr] = floatCnt;
    }
    pthread_mutex_unlock(&m_lock);
    return memptr;
  }

  void free(void *memptr) {
    pthread_mutex_lock(&m_lock);
    ensure(buffsize.count(memptr) > 0);
    unsigned int floatCnt = buffsize[memptr];
    resources[floatCnt].push_back(memptr);
    pthread_mutex_unlock(&m_lock);
  }
};


namespace image_generator {

  struct Generator {
    unordered_map<string, vector<string>> dict;
    vector<pair<CUevent, void*>> hostMemBlock;
    vector<string> keyset;
    int n_class, channel, height, width;
    pthread_spinlock_t s_lock;
    pthread_mutex_t m_lock;

    struct Worker {
      pthread_t tid;
    };

    vector<Worker> workers;
    bool threadStop;
    int cache_size;
    string format;

    MemoryManager *hostMem;

    Generator(const string &path, int height, int width, int thread_para, int batch_size, const string &format): height(height), width(width), channel(3),
        workers(thread_para), batch_size(batch_size), format(format) {
      ensure(format == "NCHW" || format == "NHWC");

      hostMem = new MemoryManager([](size_t bytes) -> void* {
        void *cudaHostPtr = nullptr;
        ensure(cudaSuccess == cudaMallocHost(&cudaHostPtr, bytes));
        return cudaHostPtr;
      }, [](void *cudaHostPtr) {
        ensure(cudaSuccess == cudaFreeHost(cudaHostPtr));
      });
      ensure(hostMem != nullptr);

      threadStop = false;
      // cache_size = (1U << 30) / thread_para / (height * width * channel * sizeof(float));
      // cache_size = max(1024, min(cache_size, (1 << 20)));
      cache_size = 16;

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
      pthread_spin_init(&s_lock, 0);
      pthread_mutex_init(&m_lock, 0);
      for (int i = 0; i < workers.size(); ++i) {
        pthread_create(&workers[i].tid, NULL, Generator::start, new pair<Generator*, int>(this, i));
      }
    }

    ~Generator() {
      for (int i = 0; i < hostMemBlock.size(); ++i) {
        auto &it = hostMemBlock[i];
        ensure(CUDA_SUCCESS == cuEventSynchronize(it.first));
        ensure(CUDA_SUCCESS == cuEventDestroy_v2(it.first));
        hostMem->free(it.second);
      }
      hostMemBlock.clear();

      threadStop = true;
      for (auto &worker: workers) {
        pthread_join(worker.tid, nullptr);
      }
      pthread_spin_destroy(&s_lock);
      pthread_mutex_destroy(&m_lock);
    }

    void background_generator(int rank) {
      unsigned int seed = (hvd_rank << 4) ^ rank;

      while (1) {
        float *cudaHostPtr = (float*)hostMem->allocate(sizeof(float) * batch_size * channel * height * width);
        int *cudaHostPtrSplit = (int*)hostMem->allocate(sizeof(int) * batch_size);

        for (int i = 0; i < batch_size; ++i) {
          float *images = cudaHostPtr + i * (channel * height * width);
          int *labels = cudaHostPtrSplit + i * (1);

          while (1) {
            int c = rand_r(&seed) % dict.size(); // rand_r(&seed)
            auto &files = dict[keyset[c]];
            if (files.size() == 0)
              continue;
            int it = rand_r(&seed) % files.size(); // rand_r(&seed)
            if (get_image_data(keyset[c] + files[it], images, c, labels))
              break;
          }
        }

        while (1) {
          if (threadStop) {
            // On Exit
            if (__sync_add_and_fetch(&activeThread, -1) == 0) {
              delete hostMem;
            }
            return;
          }
          pthread_mutex_lock(&m_lock);
          if (hostMemImages.size() >= cache_size || hostMemLabels.size() >= cache_size) {
            pthread_mutex_unlock(&m_lock);
            usleep(10000);
          } else {
            hostMemImages.push((float*)cudaHostPtr);
            hostMemLabels.push((int*)cudaHostPtrSplit);
            pthread_mutex_unlock(&m_lock);
            break;
          }
        }

      }
    }

    int batch_size;
    queue<float*> hostMemImages;
    queue<int*> hostMemLabels;

    static void *start(void *args) {
      auto data = (pair<Generator*, int>*)args;
      Generator* object = data->first;
      int rank = data->second;
      delete data;
      object->background_generator(rank);
      return NULL;
    }

    bool get_image_data(const string &image_path, float *chw, int one_hot, int *l) {
      cv::Mat image = cv::imread(image_path, 1);
      if (image.data == nullptr)
        return false;

      cv::Size dst_size(height, width);
      cv::Mat dst;
      cv::resize(image, dst, dst_size);
      *(int*)l = one_hot; // l[one_hot] = 1.0f;

      uint8_t *ptr = dst.data;

      if (format == "NCHW") {
        float *b = chw, *g = b + height * width, *r = g + height * width;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            *b++ = *ptr++ / 255.0f, *g++ = *ptr++ / 255.0f, *r++ = *ptr++ / 255.0f;
          }
        }
      } else {
        float *b = chw;
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            *b++ = *ptr++ / 255.0f, *b++ = *ptr++ / 255.0f, *b++ = *ptr++ / 255.0f;
          }
        }
      }
      return true;
    }

    vector<int> get_shape() {
      return {n_class, channel, height, width};
    }
  };
}


void *create_generator() {
  string path = getenv("GPUAAS_DATASET"), format = getenv("GPUAAS_FORMAT");
  int height = atoi(getenv("GPUAAS_HEIGHT")), width = atoi(getenv("GPUAAS_WIDTH")), batch_size = atoi(getenv("GPUAAS_BATCHSIZE"));
  int thread_para = 4;
  die_if(thread_para > 16, "Too many thread workers for image_generator: count = %d.\n", thread_para);
  if (path.size() > 0 && path[path.size() - 1] != '/')
    path += '/';
  return new image_generator::Generator(path, height, width, thread_para, batch_size, format);
}

void* next_batch(void *generator, int bytes, int type) {
  image_generator::Generator *norm_gen = (image_generator::Generator*)generator;
  auto shape = norm_gen->get_shape();
  if (type == 1) {
    ensure(bytes % (shape[1] * shape[2] * shape[3]) == 0);
    int batch_size = bytes / sizeof(float) / (shape[1] * shape[2] * shape[3]);
    ensure(norm_gen->batch_size == batch_size);

    float *front;
    while (1) {
      pthread_mutex_lock(&norm_gen->m_lock);
      if (!norm_gen->hostMemImages.size()) {
        pthread_mutex_unlock(&norm_gen->m_lock);
        continue;
      }
      front = norm_gen->hostMemImages.front();
      norm_gen->hostMemImages.pop();
      pthread_mutex_unlock(&norm_gen->m_lock);
      break;
    }
    return front;
  } else {
    ensure(type == 2);

    ensure(bytes % sizeof(float) == 0);
    int batch_size = bytes / sizeof(float);
    ensure(norm_gen->batch_size == batch_size);

    int *front;
    while (1) {
      pthread_mutex_lock(&norm_gen->m_lock);
      if (!norm_gen->hostMemLabels.size()) {
        pthread_mutex_unlock(&norm_gen->m_lock);
        continue;
      }
      front = norm_gen->hostMemLabels.front();
      norm_gen->hostMemLabels.pop();
      pthread_mutex_unlock(&norm_gen->m_lock);
      break;
    }
    return front;
  }
}

static unordered_map<CUstream, void*> dataset;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

void free_batch(void *generator, CUstream hStream, void *hostPtr) {
  image_generator::Generator *norm_gen = (image_generator::Generator*)generator;
  /*ensure(CUDA_SUCCESS == cuStreamSynchronize(hStream));
  norm_gen->hostMem->free(hostPtr);
  return;*/
  CUevent event;
  ensure(CUDA_SUCCESS == cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  ensure(CUDA_SUCCESS == cuEventRecord(event, hStream));
  // ensure(CUDA_SUCCESS == cuEventSynchronize(event));
  // ensure(CUDA_SUCCESS == cuEventDestroy_v2(event));
  pthread_spin_lock(&norm_gen->s_lock);
  norm_gen->hostMemBlock.push_back({event, hostPtr});
  for (int i = 0; i < norm_gen->hostMemBlock.size(); ++i) {
    auto &it = norm_gen->hostMemBlock[i];
    CUresult ans = cuEventQuery(it.first);
    if (ans == CUDA_SUCCESS) {
      ensure(CUDA_SUCCESS == cuEventDestroy_v2(it.first));
      norm_gen->hostMem->free(it.second);
      norm_gen->hostMemBlock[i] = norm_gen->hostMemBlock.back();
      norm_gen->hostMemBlock.pop_back();
      --i;
    } else
      ensure(ans == CUDA_ERROR_NOT_READY);
  }
  pthread_spin_unlock(&norm_gen->s_lock);
}

void free_generator(void *generator) {
  image_generator::Generator *norm_gen = (image_generator::Generator*)generator;
  delete norm_gen;
}


extern "C" CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream) {
  static int image_size = -1, label_size = -1, enable_input = -1;
  if (hvd_rank < 0) {
    const char *env = getenv("OMPI_COMM_WORLD_RANK");
    hvd_rank = env ? atoi(env) : 0;
    image_size = (env = getenv("IMAGE_SIZE")) ? atoi(env) * sizeof(float) : 0;
    label_size = (env = getenv("LABEL_SIZE")) ? atoi(env) * sizeof(float) : 0;
    enable_input = (image_size > 0);
    if (enable_input && image_size == label_size) {
      fprintf(stderr, "Low-level data input is not enabled successfully.\n");
      exit(1);
    }
  }
  // printf("rank = %d, bytes = %zd, value = %x (il = %d:%d)\n", hvd_rank, ByteCount, *(int*)srcHost, image_size, label_size);
  if (enable_input && ByteCount >= sizeof(int)) {
    pthread_mutex_lock(&g_lock);
    if (!dataset[hStream])
      dataset[hStream] = create_generator();
    void *hd = dataset[hStream];
    pthread_mutex_unlock(&g_lock);

    if (ByteCount == image_size && 0x7fc00000 == *(int*)srcHost) {
      // printf("rank = %d, bytes = %zd, value = %x (image = %d:%d)\n", hvd_rank, ByteCount, *(int*)srcHost, image_size, label_size);
      void *hostPtr = next_batch(hd, ByteCount, 1);
      ensure(cudaSuccess == cudaMemcpyAsync((void*)dstDevice, hostPtr, ByteCount, cudaMemcpyHostToDevice, hStream));
      free_batch(hd, hStream, hostPtr);
      return CUDA_SUCCESS;
    } else if (ByteCount == label_size && 0x7fc00001 == *(int*)srcHost) {
      // printf("rank = %d, bytes = %zd, value = %x (label = %d:%d)\n", hvd_rank, ByteCount, *(int*)srcHost, image_size, label_size);
      void *hostPtr = next_batch(hd, ByteCount, 2);
      ensure(cudaSuccess == cudaMemcpyAsync((void*)dstDevice, hostPtr, ByteCount, cudaMemcpyHostToDevice, hStream));
      free_batch(hd, hStream, hostPtr);
      return CUDA_SUCCESS;
    } else
      ; // printf("rank = %d, bytes = %zd, value = %x (il = %d:%d)\n", hvd_rank, ByteCount, *(int*)srcHost, image_size, label_size);
  }
  ensure(cudaSuccess == cudaMemcpyAsync((void*)dstDevice, srcHost, ByteCount, cudaMemcpyHostToDevice, hStream));
  return CUDA_SUCCESS;
}

