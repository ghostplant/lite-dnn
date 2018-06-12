/*
  mnist_mlp based on CUBLAS/CUDNN
  g++ -O3 -std=c++14 "$@" -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcblas -lcudnn

  Maintainer: Wei CUI <ghostplant@qq.com>

  Benchmark on Nvida Tesla P100:

  ----------------------------------------------------------------------------
       Model        | batch_size  |    Keras + TF_CUDA    |  Lite-DNN (C++14)
  ----------------------------------------------------------------------------
     mnist_mlp      |    32       |    8.34 sec/epoll     |  1.03 sec/epoll
     mnist_cnn      |    128      |    3.24 sec/epoll     |  1.31 sec/epoll
     cifar10_lenet  |    128      |    2.68 sec/epoll     |  1.15 sec/epoll
  ----------------------------------------------------------------------------
*/

#include <vector>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn_v7.h>

using namespace std;


#define MNIST_IMAGES "/tmp/mnist-images-idx3-ubyte"
#define MNIST_LABELS "/tmp/mnist-labels-idx1-ubyte"

#define CIFAR10_IMAGES "/tmp/cifar10-images-idx4-ubyte"
#define CIFAR10_LABELS "/tmp/cifar10-labels-idx1-ubyte"

#define TRAIN_IMAGES CIFAR10_IMAGES
#define TRAIN_LABELS CIFAR10_LABELS


static cudnnHandle_t cudnnHandle;
static cublasHandle_t cublasHandle;
static CUstream hStream = NULL;
static unsigned long lastClock;
static unordered_map<size_t, vector<void*>> cached_mem;


static inline unsigned long get_microseconds() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000LU + tv.tv_usec;
}


class DeviceMemory {
  void *d_data;
  size_t length;

public:
  DeviceMemory(size_t length): d_data(NULL), length(length) {
    if (length) {
      auto& it = cached_mem[length]; if (it.size()) { d_data = it.back(); it.pop_back(); return; }

      assert(CUDA_SUCCESS == cuMemAlloc_v2((CUdeviceptr*)&d_data, length));
    }
  }

  ~DeviceMemory() {
    if (d_data) {
      cached_mem[length].push_back(d_data); return;

      assert(CUDA_SUCCESS == cuMemFree_v2((CUdeviceptr)d_data));
    }
  }

  void* get() const {
    return d_data;
  }
};


class TensorHandler {
  cudnnTensorDescriptor_t dataTensor;

public:
  TensorHandler(const vector<int> &shape) {
    assert(shape.size() <= 4);
    int dims[4] = {1, 1, 1, 1};
    for (int i = 0; i < shape.size(); ++i)
      dims[i] = shape[i];
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateTensorDescriptor(&dataTensor));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
      dims[0], dims[1], dims[2], dims[3]));
  }

  ~TensorHandler() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));
  }

  cudnnTensorDescriptor_t get() const {
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


  Tensor(): shape({0}) {
  }

  Tensor(const vector<int> &shape, bool random = false, T range = 0) {
    size_t len = setup_tensor(shape);

    if (!random)
      return;

    auto random_uniform = [&](int size) {
      srand(size);
      vector<float> r(size);
      float avg1 = 0.0f, avg2 = 0.0f, dev;
      if (!range)
        range = sqrt(3.0 / size);

      for (int i = 0; i < r.size(); ++i) {
        r[i] = rand() / float(RAND_MAX);
        avg1 += r[i], avg2 += r[i] * r[i];
      }
      avg1 /= r.size(), avg2 /= r.size(), dev = sqrt(avg2 - avg1 * avg1);

      for (int i = 0; i < r.size(); ++i)
        r[i] = (r[i] - avg1) / dev * range;
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
    assert(CUDA_SUCCESS == cuStreamSynchronize(hStream));
  }

  vector<T> get_data() const {
    size_t len = count();
    vector<T> host(len);
    assert(CUDA_SUCCESS == cuMemcpyDtoHAsync_v2(host.data(), (CUdeviceptr)d_data->get(), len * sizeof(T), hStream));
    assert(CUDA_SUCCESS == cuStreamSynchronize(hStream));
    return move(host);
  }

  void print(bool shapeOnly = false) const {
    cout << "<< shape=(";
    for (int i = 0; i < shape.size(); ++i)
      cout << shape[i] << (i + 1 < shape.size() ? ", " : ")");
    cout << " >> ";
    if (shapeOnly) {
      cout << '\n';
      return;
    }
    vector<T> host = get_data();
    for (int i = 0; i < host.size(); ++i) {
      if (fabs(host[i]) < 1e-8)
        host[i] = 0;
      if ((i & 7) == 0)
        cout << '\n';
      cout << int(1e3 * host[i]) * 1e-3 << '\t';
      if (i + 1 == host.size())
        cout << '\n';
      if (i >= 100 && i + 1 < host.size()) {
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

  Tensor copy() const {
    Tensor<T> ans(this->shape);
    assert(CUDA_SUCCESS == cuMemcpyDtoDAsync_v2((CUdeviceptr)ans.d_data->get(), (CUdeviceptr)this->d_data->get(), ans.count() * sizeof(T), hStream));
    return move(ans);
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
    float alpha = 1.0f;
    Tensor ans(this->shape, 0.0f);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &alpha, (T*)this->d_data->get(), 1, (T*)ans.d_data->get(), 1));
    // Tensor ans = this->copy();
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &alpha, (T*)that.d_data->get(), 1, (T*)ans.d_data->get(), 1));
    return ans;
  }
};


class Layer {

public:
  virtual Tensor<float> forward(const Tensor<float> &x) = 0;

  virtual Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) = 0;

  virtual void learn(float lr) const = 0;
};


class Softmax: public Layer {

public:
  Softmax() {
  }

  ~Softmax() {
  }

  Tensor<float> forward(const Tensor<float> &x) {
    float alpha = 1.0f, beta = 0.0f;

    Tensor<float> ans(x.shape);

    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, ans.dataTensor->get(), (float*)ans.d_data->get()));
    return ans;
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    assert(x.shape == y.shape);

    float posi = 1.0f / x.shape[0], nega = -1.0f / x.shape[0], batch = 1.0f / x.shape[0];
    size_t len = x.count();

    // Tensor<float> dx = y;
    // assert(CUDNN_STATUS_SUCCESS == cudnnAddTensor(cudnnHandle,
    //   &posi, this->dataTensor->get(), (float*)this->d_data->get(), &nega, dx.dataTensor->get(), (float*)dx.d_data->get()));

    Tensor<float> dx(x.shape, 0.0f);
    cublasSaxpy(cublasHandle, len, &posi, (float*)x.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    cublasSaxpy(cublasHandle, len, &nega, (float*)y.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    return move(dx);

    /*if (lastLayer)
      return dy;

    Tensor<float> dx(x.shape, 0.0f);
    // float alpha = -float(count() / this->shape[0]) / this->shape[0], beta = 0.0f;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
      &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
      &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return dx;*/
  }

  void learn(float lr) const {
  }
};


class Activation: public Layer {
  cudnnActivationDescriptor_t activationDesc;

public:
  Activation(cudnnActivationMode_t mode) {
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateActivationDescriptor(&activationDesc));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetActivationDescriptor(activationDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
  }

  ~Activation() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyActivationDescriptor(activationDesc));
  }

  Tensor<float> forward(const Tensor<float> &x) {
    Tensor<float> ans(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationForward(cudnnHandle, activationDesc, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, ans.dataTensor->get(), (float*)ans.d_data->get()));
    return move(ans);
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    if (lastLayer)
      return dy;
    Tensor<float> dx = dy;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationBackward(cudnnHandle, activationDesc, &alpha, y.dataTensor->get(), (float*)y.d_data->get(),
        dy.dataTensor->get(), (float*)dy.d_data->get(), x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }

  void learn(float lr) const {
  }
};


class LRN: public Layer {
  unsigned lrnN;
  float bias, lrnAlpha, lrnBeta;
  cudnnLRNDescriptor_t lrnDesc;

public:
  LRN(unsigned depthRadius, float bias, float lrnAlpha, float lrnBeta):
      lrnN(2 * depthRadius + 1), bias(bias), lrnAlpha(lrnAlpha), lrnBeta(lrnBeta) {
    assert(bias >= CUDNN_LRN_MIN_K && lrnBeta >= CUDNN_LRN_MIN_BETA && lrnN >= CUDNN_LRN_MIN_N && lrnN <= CUDNN_LRN_MAX_N);
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateLRNDescriptor(&lrnDesc));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetLRNDescriptor(lrnDesc, lrnN, lrnAlpha * lrnN, lrnBeta, bias));
  }

  ~LRN() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyLRNDescriptor(lrnDesc));
  }

  Tensor<float> forward(const Tensor<float> &x) {
    assert(x.shape.size() == 4);

    Tensor<float> y(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelForward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));
    return move(y);
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    Tensor<float> dx(x.shape);
    if (lastLayer)
      return dx;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelBackward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
        x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }

  void learn(float lr) const {
  }
};


class Pooling: public Layer {
  int size, stride;
  cudnnPoolingDescriptor_t poolDesc;

public:
  Pooling(int size, int stride, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX): size(size), stride(stride) {
    assert(CUDNN_STATUS_SUCCESS == cudnnCreatePoolingDescriptor(&poolDesc));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN, size, size, 0, 0, stride, stride));
  }

  ~Pooling() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyPoolingDescriptor(poolDesc));
  }

  Tensor<float> forward(const Tensor<float> &x) {
    assert(x.shape.size() == 4);
    // assert((x.shape[2] - (size - stride)) % stride == 0 && (x.shape[3] - (size - stride)) % stride == 0);

    Tensor<float> y({x.shape[0], x.shape[1], (x.shape[2] - (size - stride)) / stride, (x.shape[3] - (size - stride)) / stride});
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));
    return move(y);
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    if (lastLayer)
      return dy;
    Tensor<float> dx(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
                         x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }

  void learn(float lr) const {
  }
};


class Dropout: public Layer {
  cudnnDropoutDescriptor_t dropDesc;

  shared_ptr<DeviceMemory> states, reversed;
  size_t states_size, reversed_size;
  uint64_t seed; float drop_prob;

public:
  Dropout(float drop_prob = 0.1f, uint64_t seed = 10): reversed_size(~0LU), seed(seed), drop_prob(drop_prob) {
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateDropoutDescriptor(&dropDesc));
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutGetStatesSize(cudnnHandle, &states_size));

    states = make_shared<DeviceMemory>(states_size);
    assert(CUDNN_STATUS_SUCCESS == cudnnSetDropoutDescriptor(dropDesc, cudnnHandle, drop_prob, states->get(), states_size, seed));
  }

  ~Dropout() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyDropoutDescriptor(dropDesc));
  }

  Tensor<float> forward(const Tensor<float> &x) {
    size_t _reversed_size;
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(x.dataTensor->get(), &_reversed_size));
    if (~reversed_size)
      assert(_reversed_size == reversed_size);
    else {
      reversed_size = _reversed_size;
      reversed = make_shared<DeviceMemory>(states_size);
    }

    Tensor<float> ans(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutForward(cudnnHandle, dropDesc, x.dataTensor->get(), (float*)x.d_data->get(),
      ans.dataTensor->get(), (float*)ans.d_data->get(), reversed->get(), reversed_size));
    return move(ans);
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    assert(CUDNN_STATUS_SUCCESS == cudnnRestoreDropoutDescriptor(dropDesc, cudnnHandle, drop_prob, states->get(), states_size, seed));
    size_t _reversed_size;
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(y.dataTensor->get(), &_reversed_size));
    assert(_reversed_size == reversed_size);

    if (lastLayer)
      return dy;
    Tensor<float> dx(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutBackward(cudnnHandle, dropDesc, dy.dataTensor->get(), (float*)dy.d_data->get(),
      dx.dataTensor->get(), (float*)dx.d_data->get(), reversed->get(), reversed_size));
    return move(dx);
  }

  void learn(float lr) const {
  }
};


class Dense: public Layer {
  Tensor<float> w, bias, ones, g_bias, g_w;
  int channels;

public:
  Dense(int channels, int max_batch = 1024): channels(channels), bias({1, channels}, true), ones({max_batch, 1}, 1.0f), w(), g_bias(), g_w() {
  }

  Tensor<float> forward(const Tensor<float> &x) {
    if (w.count() < 1) {
      w = Tensor<float>({channels, x.shape[1]}, true);
      g_w = Tensor<float>(w.shape);
      g_bias = Tensor<float>(bias.shape);
    }

    auto out = x.matmul(w, false, true);
    // out = x * w' + ones * bias';
    assert(out.shape.size() == 2 && bias.shape.size() == 2 && out.shape[1] == bias.shape[1] && out.shape[0] <= ones.shape[0]);

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

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    g_w = dy.matmul(x, true, false);
    g_bias = ones.reshape({x.shape[0], 1}, true).matmul(dy, true, false);
    if (lastLayer)
      return dy;
    return dy.matmul(w);
  }

  void learn(float lr) const {
    assert(w.shape == g_w.shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, w.count(), &lr, (float*)g_w.d_data->get(), 1, (float*)w.d_data->get(), 1));

    assert(bias.shape == g_bias.shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, bias.count(), &lr, (float*)g_bias.d_data->get(), 1, (float*)bias.d_data->get(), 1));
  }
};


class Flatten: public Layer {

public:
  Flatten() {
  }

  Tensor<float> forward(const Tensor<float> &x) {
    return x.reshape({x.shape[0], int(x.count() / x.shape[0])});
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    return dy.reshape(x.shape);
  }

  void learn(float lr) const {
  }
};


class Convolution: public Layer {
  int filters, kernel_size;
  Tensor<float> w_krnl, w_bias, g_krnl, g_bias;
  bool use_bias;

  cudnnConvolutionDescriptor_t convDesc;
  cudnnFilterDescriptor_t filterDesc;

public:
  Convolution(int filters, int kernel_size, bool use_bias = false): w_krnl(), w_bias(), g_krnl(), g_bias(),
      filters(filters), kernel_size(kernel_size), convDesc(NULL), filterDesc(NULL), use_bias(use_bias) {
  }

  void configure(int in_chans) {
    w_krnl = Tensor<float>({in_chans * filters * kernel_size * kernel_size}, true);
    g_krnl = Tensor<float>(w_krnl.shape);
    if (use_bias) {
      w_bias = Tensor<float>({1, filters, 1, 1}, true);
      g_bias = Tensor<float>(w_bias.shape);
    }

    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    // assert(CUDNN_STATUS_SUCCESS == cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

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
    if (w_krnl.count() < 1)
      configure(x.shape[1]);
    assert(x.shape.size() == 4);

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

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) {
    float alpha = 1.0f, beta = 0.0f;
    assert(y.shape[1] == filters);
    int n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];

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
                filterDesc, (float*)g_krnl.d_data->get()));

    if (use_bias) {
        assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardBias(cudnnHandle, &alpha,
                dy.dataTensor->get(), (float*)dy.d_data->get(), &beta,
                g_bias.dataTensor->get(), (float*)g_bias.d_data->get()));
    }

    if (lastLayer)
      return dy;

    Tensor<float> dx({n, c, h, w});
    assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardData(cudnnHandle, &alpha,
                filterDesc, (float*)w_krnl.d_data->get(),
                dy.dataTensor->get(), (float*)dy.d_data->get(),
                convDesc, dalgo, workspace.get(), maxSizeInBytes, &beta,
                dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }

  void learn(float lr) const {
    assert(w_krnl.shape == g_krnl.shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, w_krnl.count(), &lr, (float*)g_krnl.d_data->get(), 1, (float*)w_krnl.d_data->get(), 1));

    if (use_bias) {
      assert(w_bias.shape == g_bias.shape);
      assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, w_bias.count(), &lr, (float*)g_bias.d_data->get(), 1, (float*)w_bias.d_data->get(), 1));
    }
  }
};


static pair<vector<int>, vector<float>> ReadNormalDataset(const char* dataset) {
  auto read_uint32 = [&](FILE *fp) {
    uint32_t val;
    assert(fread(&val, sizeof(val), 1, fp) == 1);
    return __builtin_bswap32(val);
  };

  const int UBYTE_MAGIC = 0x800;
  FILE *fp;
  assert((fp = fopen(dataset, "rb")) != NULL);

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
}


vector<shared_ptr<Layer>> create_model() {
  vector<shared_ptr<Layer>> layers;
  /* CIFAR10_ALEXNET
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
  */

  // CIFAR10_LENET
  layers.push_back(make_shared<Convolution>(32, 5, true));
  layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
  layers.push_back(make_shared<Convolution>(64, 5, true));
  layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
  layers.push_back(make_shared<Flatten>());
  layers.push_back(make_shared<Dense>(512));
  layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
  layers.push_back(make_shared<Dense>(10));
  layers.push_back(make_shared<Softmax>());

  /* MNIST_MLP
  layers.push_back(make_shared<Flatten>());
  layers.push_back(make_shared<Dense>(512));
  layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
  layers.push_back(make_shared<Dropout>(0.1));
  layers.push_back(make_shared<Dense>(512));
  layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
  layers.push_back(make_shared<Dropout>(0.1));
  layers.push_back(make_shared<Dense>(10));
  layers.push_back(make_shared<Softmax>());
  */

  /* MNIST_CNN
  layers.push_back(make_shared<Convolution>(32, 3));
  layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
  layers.push_back(make_shared<Convolution>(64, 3));
  layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
  layers.push_back(make_shared<Pooling>(2, 2, CUDNN_POOLING_MAX));
  layers.push_back(make_shared<Flatten>());
  layers.push_back(make_shared<Dense>(128));
  layers.push_back(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
  layers.push_back(make_shared<Dense>(10));
  layers.push_back(make_shared<Softmax>());
  */
  return move(layers);
}

int main() {
  Tensor<float>::init();

  auto full_images = ReadNormalDataset(TRAIN_IMAGES);
  auto full_labels = ReadNormalDataset(TRAIN_LABELS);
  assert(full_images.first[0] == full_labels.first[0]);

  int samples = full_images.first[0];
  int width = full_images.first[1] * full_images.first[2] * full_images.first[3];
  int classes = full_labels.first[1];
  printf("Total %d samples (%d, %d, %d) for %d classes found.\n", samples, full_images.first[1], full_images.first[2], full_images.first[3], classes);

  auto model = create_model();

  vector<Tensor<float>> input(model.size() + 1), dloss(model.size() + 1);
  int batch_size = 128, epochs = 100, steps = (samples + batch_size - 1) / batch_size * epochs;
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

    dloss[model.size()] = model.back()->backward(input.back(), labels, input.back());
    for (int i = model.size() - 1; i >= 1; --i)
      dloss[i] = model[i - 1]->backward(dloss[i + 1], input[i], input[i - 1], i == 1), model[i - 1]->learn(lr);
    auto data_output = input.back(), data_loss = dloss.back();

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

