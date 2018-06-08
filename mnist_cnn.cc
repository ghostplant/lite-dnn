/*
  mnist_mlp based on CUBLAS/CUDNN
  g++ -O3 -std=c++14 "$@" -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcblas -lcudnn

  Maintainer: Wei CUI <ghostplant@qq.com>

  Benchmark on Nvida Tesla P100:

  --------------------------------------------------------------------------
       Model      | batch_size  |    Keras + TF_CUDA    |  Lite-DNN (C++14)
  --------------------------------------------------------------------------
     mnist_mlp    |    32       |    8.34 sec/epoll     |  2.76 sec/epoll
     mnist_cnn    |    128      |    3.24 sec/epoll     |  1.29 sec/epoll
  --------------------------------------------------------------------------
*/

#include <vector>
#include <iostream>
#include <memory>
#include <random>

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn_v7.h>

using namespace std;

#define MNIST_IMAGES "/tmp/train-images-idx3-ubyte"
#define MNIST_LABELS "/tmp/train-labels-idx1-ubyte"

#define  SLOT_COUNT   29
#define  LOG2(X)      (debruijn[((uint32_t)(((X) & -(X)) * 0x077CB531U)) >> 27])

static cudnnHandle_t cudnnHandle;
static cublasHandle_t cublasHandle;
static CUstream hStream = NULL;
static unsigned long lastClock;
static vector<void*> block[SLOT_COUNT];

static inline int get_slot(ssize_t bytes) {
  static const int debruijn[32] = {0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9};

  bytes += sizeof(ssize_t) - 1;
  bytes = bytes | (bytes >> 1);
  bytes = bytes | (bytes >> 2);
  bytes = bytes | (bytes >> 4);
  bytes = bytes | (bytes >> 8);
  bytes = bytes | (bytes >> 16);
  ++bytes;

  if (bytes < (1LU << SLOT_COUNT))
    return LOG2(bytes);
  return -1;
}

static inline unsigned long get_microseconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000LU + tv.tv_usec;
}

static pair<vector<uint8_t>, vector<uint8_t> > ReadUByteDataset(const char* image_filename, const char* label_filename) {
  auto read_uint32 = [&](FILE *fp) {
    uint32_t val;
    assert(fread(&val, sizeof(val), 1, fp) == 1);
    return __builtin_bswap32(val);
  };

  const int UBYTE_MAGIC = 0x800;
  FILE *fp;
  uint32_t header, length, h, w;
  
  assert((fp = fopen(image_filename, "rb")) != NULL);
  header = read_uint32(fp);
  length = read_uint32(fp);
  h = read_uint32(fp);
  w = read_uint32(fp);

  pair<vector<uint8_t>, vector<uint8_t> > ans;
  ans.first.resize(length * w * h);
  ans.second.resize(length);

  assert(header == UBYTE_MAGIC + 3);
  assert(fread(ans.first.data(), 1, ans.first.size(), fp) == ans.first.size());
  fclose(fp);

  assert((fp = fopen(label_filename, "rb")) != NULL);
  header = read_uint32(fp);
  length = read_uint32(fp);

  assert(header == UBYTE_MAGIC + 1);
  assert(fread(ans.second.data(), 1, ans.second.size(), fp) == ans.second.size());
  fclose(fp);

  return move(ans);
}


class DeviceMemory {
  void *d_data;
  int slot;

public:
  DeviceMemory(size_t length): d_data(NULL) {
    if (length) {
      slot = get_slot(length);
      if (slot >= 0 && block[slot].size() > 0) {
        void *ptr = block[slot].back();
        block[slot].pop_back();
        d_data = ptr;
      } else
        assert(CUDA_SUCCESS == cuMemAlloc_v2((CUdeviceptr*)&d_data, length));
    }
  }

  ~DeviceMemory() {
    if (d_data) {
      if (slot >= 0)
        block[slot].push_back(d_data);
      else
        assert(CUDA_SUCCESS == cuMemFree_v2((CUdeviceptr)d_data));
    }
  }

  void* get() const {
    return d_data;
  }
};

class TensorHandler {
  cudnnTensorDescriptor_t dataTensor;
  vector<int> shape;

public:
  TensorHandler(const vector<int> &shape): shape(shape), dataTensor(NULL) {
    assert(shape.size() <= 4);
    while (this->shape.size() < 4)
      this->shape.push_back(1);
  }

  ~TensorHandler() {
    if (dataTensor)
      assert(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));
  }

  cudnnTensorDescriptor_t get() {
    if (!dataTensor) {
      assert(CUDNN_STATUS_SUCCESS == cudnnCreateTensorDescriptor(&dataTensor));
      assert(CUDNN_STATUS_SUCCESS == cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1], shape[2], shape[3]));
    }
    return dataTensor;
  }
};


template <class T> class Tensor {

public:
  shared_ptr<DeviceMemory> d_data;
  shared_ptr<TensorHandler> dataTensor;
  vector<int> shape;

  static void init(bool randTime = false) {
    int devCount = 0;
    CUcontext primaryCtx;

    cuInit(0);
    cuDeviceGetCount(&devCount);
    assert(devCount > 0);
    cuDevicePrimaryCtxRetain(&primaryCtx, 0);
    cuCtxSetCurrent(primaryCtx);
    assert(CUDA_SUCCESS == cuStreamCreate(&hStream, 0));

    cublasCreate(&cublasHandle);
    cublasSetStream_v2(cublasHandle, hStream);
    cublasSetPointerMode_v2(cublasHandle, CUBLAS_POINTER_MODE_HOST);

    cudnnCreate(&cudnnHandle);
    cudnnSetStream(cudnnHandle, hStream);

    if (randTime)
      srand(time(0));

    lastClock = get_microseconds();
  }


  Tensor(const vector<int> &shape, bool random = false) {
    size_t len = setup_tensor(shape);

    if (!random)
      return;

    float range = sqrt(3.0f / len);

    auto random_uniform = [&](int size) {
      vector<float> r(size);
      float avg1 = 0.0f, avg2 = 0.0f, dev;

      for (int i = 0; i < r.size(); ++i) {
        r[i] = rand() / float(RAND_MAX);
        avg1 += r[i], avg2 += r[i] * r[i];
      }
      avg1 /= r.size(), avg2 /= r.size(), dev = sqrt(avg2 - avg1 * avg1);

      for (int i = 0; i < r.size(); ++i)
        r[i] = (r[i] - avg1) * range / dev;
      return move(r);
    };

    set_data(random_uniform(len));
  }

  Tensor(const vector<int> &shape, const vector<T> &host) {
    size_t len = setup_tensor(shape);

    assert(host.size() == len);
    set_data(host);
  }

  Tensor(const vector<int> &shape, const T val) {
    size_t len = setup_tensor(shape);

    assert(sizeof(T) == sizeof(unsigned int));
    unsigned int ui = (unsigned int&)val;
    assert(CUDA_SUCCESS == cuMemsetD32Async((CUdeviceptr)d_data->get(), ui, len, hStream));
  }


  size_t setup_tensor(const vector<int> &shape) {
    this->shape = shape;
    size_t len = count();
    d_data = make_shared<DeviceMemory>(len * sizeof(T));
    dataTensor = make_shared<TensorHandler>(shape);
    return len;
  }

  size_t count() const {
    size_t len = 1;
    for (auto it: shape)
      len *= it;
    return len;
  }

  void set_data(const vector<T> &host) const {
    size_t len = count();
    assert(len == host.size());
    assert(CUDA_SUCCESS == cuMemcpyHtoDAsync_v2((CUdeviceptr)d_data->get(), host.data(), len * sizeof(T), hStream));
  }

  vector<T> get_data() const {
    size_t len = count();
    vector<T> host(len);
    assert(CUDA_SUCCESS == cuMemcpyDtoHAsync_v2(host.data(), (CUdeviceptr)d_data->get(), len * sizeof(T), hStream));
    assert(CUDA_SUCCESS == cuStreamSynchronize(hStream));
    return move(host);
  }

  void print(bool shapeOnly = false) const {
    vector<T> host = get_data();
    cout << "<< shape=(";
    for (int i = 0; i < shape.size(); ++i)
      cout << shape[i] << (i + 1 < shape.size() ? ", " : ")");
    cout << " >> ";
    if (shapeOnly) {
      cout << '\n';
      return;
    }
    for (int i = 0; i < host.size(); ++i) {
      if (fabs(host[i]) < 1e-8)
        host[i] = 0;
      cout << host[i] << (i + 1 < host.size() ? ' ' : '\n');
      if (i >= 32 && i + 1 < host.size()) {
        cout << "...\n";
        break;
      }
    }
  }

  Tensor reshape(const vector<int> &shape, bool weak = false) const {
    Tensor mat = *this;
    mat.shape = shape;
    mat.dataTensor = make_shared<TensorHandler>(shape);
    if (!weak)
      assert(mat.count() == count());
    return move(mat);
  }

  Tensor matmul(const Tensor<T> &that, bool transposeThis = false, bool transposeThat = false) const {
    // ans = &that * this;
    const Tensor<T> *A = &that, *B = this;
    bool transposeA = transposeThat, transposeB = transposeThis;

    assert(A->shape.size() == 2 && B->shape.size() == 2);

    int ax = A->shape[1], ay = A->shape[0];
    if (transposeA)
      swap(ax, ay);
    int bx = B->shape[1], by = B->shape[0];
    if (transposeB)
      swap(bx, by);
    assert(ay == bx);

    Tensor<T> ans({by, ax});

    float alpha = 1.0f, beta = 0.0f;
    assert(0 == cublasSgemm(cublasHandle,
                            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                            ax, by, ay, &alpha,
                            (T*)A->d_data->get(), A->shape[1], // X
                            (T*)B->d_data->get(), B->shape[1],  // Y
                            &beta, (T*)ans.d_data->get(), ans.shape[1]));   // Z
    return ans;
  }

  Tensor matadd(const Tensor<T> &that) const {
    assert(this->shape == that.shape);
    Tensor ans(this->shape, 0);
    float alpha = 1.0f;
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &alpha, (T*)this->d_data->get(), 1, (T*)ans.d_data->get(), 1));
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &alpha, (T*)that.d_data->get(), 1, (T*)ans.d_data->get(), 1));
    return ans;
  }

  Tensor softmaxLoss(Tensor<T> &outputs) const {
    assert(this->shape.size() == 2 && this->shape == outputs.shape);

    float posi = 1.0f / this->shape[0], nega = -1.0f / this->shape[0], batch = 1.0f / this->shape[0];
    Tensor<T> loss = outputs;
    assert(CUDNN_STATUS_SUCCESS == cudnnAddTensor(cudnnHandle,
      &posi, this->dataTensor->get(), (float*)this->d_data->get(), &nega, loss.dataTensor->get(), (float*)loss.d_data->get()));
    return loss;

    Tensor<T> ans(this->shape, 0.0f);

    cublasSaxpy(cublasHandle, count(), &posi, (T*)this->d_data->get(), 1, (T*)ans.d_data->get(), 1);
    cublasSaxpy(cublasHandle, count(), &nega, (T*)outputs.d_data->get(), 1, (T*)ans.d_data->get(), 1);
    return ans;

    float alpha = -float(count() / this->shape[0]) / this->shape[0], beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
      &alpha, this->dataTensor->get(), (T*)this->d_data->get(), outputs.dataTensor->get(), (T*)outputs.d_data->get(),
      &beta, ans.dataTensor->get(), (T*)ans.d_data->get()));
    return ans;
  }

  Tensor softmaxForward() const {
    float alpha = 1.0f, beta = 0.0f;

    Tensor<T> ans(this->shape);

    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha, dataTensor->get(), (T*)this->d_data->get(), &beta, dataTensor->get(), (T*)ans.d_data->get()));
    return ans;
  }

  Tensor lrnCrossForward(unsigned depthRadius, float bias, float lrnAlpha, float lrnBeta) const {
    assert(this->shape.size() == 4);

    float alpha = 1.0f, beta = 0.0f;

    Tensor<T> ans(this->shape);

    unsigned lrnN = 2 * depthRadius + 1;
    assert(bias > 1e-5);
    cudnnLRNDescriptor_t lrnDesc;
    cudnnCreateLRNDescriptor(&lrnDesc);
    assert(CUDNN_STATUS_SUCCESS == cudnnSetLRNDescriptor(lrnDesc, lrnN, lrnAlpha * lrnN, lrnBeta, bias));
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelForward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, this->dataTensor->get(), (T*)this->d_data->get(), &beta, ans.dataTensor->get(), (T*)ans.d_data->get()));
    cudnnDestroyLRNDescriptor(lrnDesc);
    return ans;
  }

  Tensor lrnCrossBackward(const Tensor<T> &tensorPost, const Tensor<T> &tensorPre, unsigned depthRadius, float bias, float lrnAlpha, float lrnBeta) const {
    assert(this->shape.size() == 4);

    float alpha = 1.0f, beta = 0.0f;

    Tensor<T> ans(this->shape);

    unsigned lrnN = 2 * depthRadius + 1;
    assert(bias >= CUDNN_LRN_MIN_K && lrnBeta >= CUDNN_LRN_MIN_BETA && lrnN >= CUDNN_LRN_MIN_N && lrnN <= CUDNN_LRN_MAX_N);
    cudnnLRNDescriptor_t lrnDesc;
    cudnnCreateLRNDescriptor(&lrnDesc);
    assert(CUDNN_STATUS_SUCCESS == cudnnSetLRNDescriptor(lrnDesc, lrnN, lrnAlpha * lrnN, lrnBeta, bias));
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelBackward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, tensorPost.dataTensor->get(), (T*)tensorPost.d_data->get(), this->dataTensor->get(), (T*)this->d_data->get(),
        &beta, tensorPre.dataTensor->get(), (T*)tensorPre.d_data->get(), ans.dataTensor->get(), (T*)ans.d_data->get()));
    cudnnDestroyLRNDescriptor(lrnDesc);
    return ans;
  }
};


class Activation {
  cudnnActivationDescriptor_t activationDesc;

public:
  Activation(cudnnActivationMode_t mode) {
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateActivationDescriptor(&activationDesc));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetActivationDescriptor(activationDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
  }

  ~Activation() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyActivationDescriptor(activationDesc));
  }

  Tensor<float> forward(const Tensor<float> &x) const {
    Tensor<float> ans(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationForward(cudnnHandle, activationDesc, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, ans.dataTensor->get(), (float*)ans.d_data->get()));
    return move(ans);
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) const {
    Tensor<float> dx = dy;
    if (lastLayer)
      return dx;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationBackward(cudnnHandle, activationDesc, &alpha, y.dataTensor->get(), (float*)y.d_data->get(),
        dy.dataTensor->get(), (float*)dy.d_data->get(), x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }
};


class Pooling {
  int stride;
  cudnnPoolingDescriptor_t poolDesc;

public:
  Pooling(int size, int stride, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX): stride(stride) {
    assert(CUDNN_STATUS_SUCCESS == cudnnCreatePoolingDescriptor(&poolDesc));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN, size, size, 0, 0, stride, stride));
  }

  ~Pooling() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyPoolingDescriptor(poolDesc));
  }

  Tensor<float> forward(const Tensor<float> &x) const {
    assert(x.shape.size() == 4);

    Tensor<float> ans({x.shape[0], x.shape[1], x.shape[2] / stride, x.shape[3] / stride});
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, ans.dataTensor->get(), (float*)ans.d_data->get()));
    return move(ans);
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) const {
    Tensor<float> dx(x.shape);
    if (lastLayer)
      return dx;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
                         x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }
};


class Dense {
  Tensor<float> w, bias, ones;
  int channels;

public:
  Dense(int channels, int max_batch = 1024): channels(channels), w({1, 1}), bias({1, channels}, true), ones({max_batch, 1}, 1.0f) {
  }

  Tensor<float> forward(const Tensor<float> &x) {
    if (w.count() <= 1)
      w = Tensor<float>({channels, x.shape[1]}, true);

    auto out = x.matmul(w, false, true);
    // out = x * w' + ones * bias';
    assert(out.shape.size() == 2 && bias.shape.size() == 2 && out.shape[1] == bias.shape[1] && out.shape[0] <= ones.shape[0]);
    // return out.matadd(ones.matmul(bias));

    float alpha = 1.0f;
    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                bias.shape[1], out.shape[0], 1,
                &alpha,
                (float*)bias.d_data->get(), bias.shape[1],  // B
                (float*)ones.d_data->get(), 1,  // 1
                &alpha,
                (float*)out.d_data->get(), out.shape[1]);  // self
    return move(out);
  }

  vector<Tensor<float> > backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    vector<Tensor<float> > tensors;
    tensors.push_back(dy.matmul(x, true, false));
    tensors.push_back(ones.reshape({x.shape[0], 1}, true).matmul(dy, true, false));
    if (!lastLayer)
      tensors.push_back(dy.matmul(w));
    return move(tensors);
  }

  void learn(const vector<Tensor<float> > &tensors, float lr = -0.05) const {
    assert(w.shape == tensors[0].shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, w.count(), &lr, (float*)tensors[0].d_data->get(), 1, (float*)w.d_data->get(), 1));

    assert(bias.shape == tensors[1].shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, bias.count(), &lr, (float*)tensors[1].d_data->get(), 1, (float*)bias.d_data->get(), 1));
  }
};


class Flatten {

public:
  Flatten() {
  }

  Tensor<float> forward(const Tensor<float> &x) {
    return x.reshape({x.shape[0], int(x.count() / x.shape[0])});
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    return dy.reshape(x.shape);
  }
};


class Convolution {
  int filters, in_chans, kernel_size;
  Tensor<float> w_krnl, w_bias;
  bool use_bias;

  cudnnConvolutionDescriptor_t convDesc;
  cudnnFilterDescriptor_t filterDesc;

public:
  Convolution(int filters, int kernel_size, bool use_bias = false): w_krnl({1, 1}), w_bias({1, 1}),
      filters(filters), kernel_size(kernel_size), in_chans(-1), convDesc(NULL), filterDesc(NULL), use_bias(use_bias) {
  }

  void configure(int in_chans) {
    this->in_chans = in_chans;
    w_krnl = Tensor<float>({in_chans * filters * kernel_size * kernel_size}, true);
    if (use_bias)
      w_bias = Tensor<float>({1, filters, 1, 1}, true);

    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    assert(CUDNN_STATUS_SUCCESS == cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filters, in_chans, kernel_size, kernel_size);
  }

  ~Convolution() {
    if (filterDesc)
      assert(CUDNN_STATUS_SUCCESS == cudnnDestroyFilterDescriptor(filterDesc));
    if (convDesc)
      assert(CUDNN_STATUS_SUCCESS == cudnnDestroyConvolutionDescriptor(convDesc));
  }

  Tensor<float> forward(const Tensor<float> &x) {
    if (in_chans < 0)
      configure(x.shape[1]);
    assert(x.shape.size() == 4 && x.shape[1] == in_chans);

    float alpha = 1.0f, beta = 0.0f;
    int n = x.shape[0], c = filters, h = x.shape[2] - (kernel_size - 1), w = x.shape[3] - (kernel_size - 1);
    // assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolution2dForwardOutputDim(convDesc, x.dataTensor->get(), filterDesc, &n, &c, &h, &w));

    Tensor<float> ans({n, c, h, w});

    cudnnConvolutionFwdAlgo_t convalgo;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardAlgorithm(cudnnHandle, x.dataTensor->get(), filterDesc, convDesc,
        ans.dataTensor->get(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convalgo));

    size_t sizeInBytes;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, x.dataTensor->get(), filterDesc, convDesc,
        ans.dataTensor->get(), convalgo, &sizeInBytes));

    DeviceMemory workspace(sizeInBytes);

    assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionForward(cudnnHandle, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), filterDesc,
        (float*)w_krnl.d_data->get(), convDesc, convalgo, workspace.get(), sizeInBytes, &beta, ans.dataTensor->get(), (float*)ans.d_data->get()));
    if (use_bias)
      assert(CUDNN_STATUS_SUCCESS == cudnnAddTensor(cudnnHandle,
        &alpha, w_bias.dataTensor->get(), (float*)w_bias.d_data->get(), &alpha, ans.dataTensor->get(), (float*)ans.d_data->get()));
    return move(ans);
  }

  vector<Tensor<float> > backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) const {
    float alpha = 1.0f, beta = 0.0f;
    assert(y.shape[1] == filters);
    int n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];

    Tensor<float> grad(w_krnl.shape), dx({n, c, h, w});

    cudnnConvolutionBwdFilterAlgo_t falgo;
    cudnnConvolutionBwdDataAlgo_t dalgo;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle, x.dataTensor->get(), y.dataTensor->get(), convDesc, filterDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &falgo));
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle, filterDesc, y.dataTensor->get(), convDesc, x.dataTensor->get(),
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &dalgo));

    size_t maxSizeInBytes = 0, sizeInBytes;

    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnnHandle, x.dataTensor->get(), y.dataTensor->get(), convDesc, filterDesc, 
                falgo, &sizeInBytes));
    maxSizeInBytes = max(maxSizeInBytes, sizeInBytes);
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle, filterDesc, y.dataTensor->get(), convDesc, x.dataTensor->get(), 
                dalgo, &sizeInBytes));
    maxSizeInBytes = max(maxSizeInBytes, sizeInBytes);

    DeviceMemory workspace(maxSizeInBytes);

    assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardFilter(cudnnHandle, &alpha,
                x.dataTensor->get(), (float*)x.d_data->get(),
                dy.dataTensor->get(), (float*)dy.d_data->get(),
                convDesc, falgo, workspace.get(), maxSizeInBytes, &beta,
                filterDesc, (float*)grad.d_data->get()));

    vector<Tensor<float> > tensors;
    tensors.push_back(move(grad));

    if (use_bias) {
        Tensor<float> bias(w_bias.shape);
        assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardBias(cudnnHandle, &alpha,
                dy.dataTensor->get(), (float*)dy.d_data->get(), &beta,
                bias.dataTensor->get(), (float*)bias.d_data->get()));
        tensors.push_back(move(bias));
    }

    if (!lastLayer) {
      assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardData(cudnnHandle, &alpha,
                filterDesc, (float*)w_krnl.d_data->get(),
                dy.dataTensor->get(), (float*)dy.d_data->get(),
                convDesc, dalgo, workspace.get(), maxSizeInBytes, &beta,
                dx.dataTensor->get(), (float*)dx.d_data->get()));
      tensors.push_back(move(dx));
    }
    return move(tensors);
  }

  void learn(const vector<Tensor<float> > &tensors, float lr = -0.05) const {
    assert(w_krnl.shape == tensors[0].shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, w_krnl.count(), &lr, (float*)tensors[0].d_data->get(), 1, (float*)w_krnl.d_data->get(), 1));

    if (use_bias) {
      assert(w_bias.shape == tensors[1].shape);
      assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, w_bias.count(), &lr, (float*)tensors[1].d_data->get(), 1, (float*)w_bias.d_data->get(), 1));
    }
  }
};


int main() {
  Tensor<float>::init();

  Dense fc1(512), fc2(512), fc3(10);
  Flatten flatten;

  Dense dense1(128), dense2(10);
  Activation relu(CUDNN_ACTIVATION_RELU);
  Pooling pooling(2, 2, CUDNN_POOLING_MAX);
  Convolution cnn1(32, 3), cnn2(64, 3);

  auto dataset = ReadUByteDataset(MNIST_IMAGES, MNIST_LABELS);
  int samples = dataset.first.size() / 784;
  printf("Total %d samples found.\n", samples);

  int batch_size = 128;
  int it = 0, epochs = 100, steps = (samples + batch_size - 1) / batch_size * epochs;
  for (int k = 0; k < steps; ++k) {
    vector<float> in(784 * batch_size), out(10 * batch_size);
    memset(out.data(), 0, out.size() * sizeof(float));
    for (int i = 0; i < batch_size; ++i, it = (it + 1) % samples) {
      int lb = dataset.second[it * 1];
      out[i * 10 + lb] = 1.0f;
      for (int j = 0; j < 784; ++j)
        in[i * 784 + j] = dataset.first[it * 784 + j] / 255.0;
    }
    Tensor<float> images({batch_size, 1, 28, 28}, in), labels({batch_size, 10}, out);

    // << MNIST_CNN >>
    auto y0 = images;
    auto y1 = cnn1.forward(y0);
    auto y2 = relu.forward(y1);
    auto y3 = cnn2.forward(y2);
    auto y4 = relu.forward(y3);
    auto y5 = pooling.forward(y4);
    auto y6 = flatten.forward(y5);
    auto y7 = dense1.forward(y6);
    auto y8 = relu.forward(y7);
    auto y9 = dense2.forward(y8);
    auto y10 = y9.softmaxForward();

    auto dy9 = y10.softmaxLoss(labels);
    auto dy8pack = dense2.backward(dy9, y9, y8); dense2.learn(dy8pack); auto dy8 = dy8pack.back();
    auto dy7 = relu.backward(dy8, y8, y7);
    auto dy6pack = dense1.backward(dy7, y7, y6); dense1.learn(dy6pack); auto dy6 = dy6pack.back();
    auto dy5 = flatten.backward(dy6, y6, y5);
    auto dy4 = pooling.backward(dy5, y5, y4);
    auto dy3 = relu.backward(dy4, y4, y3);
    auto dy2pack = cnn2.backward(dy3, y3, y2); cnn2.learn(dy2pack); auto dy2 = dy2pack.back();
    auto dy1 = relu.backward(dy2, y2, y1);
    auto dy0pack = cnn1.backward(dy1, y1, y0, true); cnn1.learn(dy0pack); // auto dy0 = dy0pack.back();

    auto data_output = y9, data_loss = y10;


    /* << MNIST_MLP >>
    auto y0 = pooling.forward(images).reshape({batch_size, 784 / 4}); // images.reshape({batch_size, 784});
    auto y1 = fc1.forward(y0);
    auto y2 = relu.forward(y1);
    auto y3 = dense1.forward(y2);
    auto y4 = relu.forward(y3);
    auto y5 = dense2.forward(y4);
    auto y6 = y5.softmaxForward();

    auto dy5 = y6.softmaxLoss(labels);
    auto dy4pack = dense2.backward(dy5, y4, y4); dense2.learn(dy4pack); auto dy4 = dy4pack.back();
    auto dy3 = relu.backward(dy4, y4, y3);
    auto dy2pack = dense1.backward(dy3, y3, y2); dense1.learn(dy2pack); auto dy2 = dy2pack.back();
    auto dy1 = relu.backward(dy2, y2, y1);
    auto dy0pack = fc1.backward(dy1, y1, y0); fc1.learn(dy0pack); auto dy0 = dy0pack.back();

    auto data_output = y5, data_loss = y6;
    */

    if (it < batch_size) {
      vector<float> loss_data = data_loss.get_data();
      float loss = 0.0f;
      for (int i = 0; i < loss_data.size(); ++i) {
        float j = fabs(loss_data[i]);
        if (j >= 1e-8)
          loss += -j * log(j);
      }
      loss /= data_loss.shape[0];

      vector<float> pred_data = data_output.get_data();
      int tot = 0, acc = 0;
      for (int i = 0; i < batch_size; ++i) {
        int it = 0, jt = 0;
        for (int j = 1; j < 10; ++j) {
          if (pred_data[i * 10 + it] < pred_data[i * 10 + j])
            it = j;
          if (out[i * 10 + jt] < out[i * 10 + j])
            jt = j;
        }
        ++tot;
        if (it == jt)
          ++acc;
      }
      static int epoch = 0;

      unsigned long currClock = get_microseconds();
      printf("epoch = %d: loss = %.4f, acc = %.2f%%, time = %.4fs\n", ++epoch, loss, acc * 100.0f / tot, (currClock - lastClock) * 1e-6f);
      lastClock = currClock;
    }
  }
  return 0;
}

