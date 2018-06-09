/*
  mnist_mlp based on CUBLAS/CUDNN
  g++ -O3 -std=c++14 "$@" -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcblas -lcudnn

  Maintainer: Wei CUI <ghostplant@qq.com>

  Benchmark on Nvida Tesla P100:

  ----------------------------------------------------------------------------
       Model        | batch_size  |    Keras + TF_CUDA    |  Lite-DNN (C++14)
  ----------------------------------------------------------------------------
     mnist_mlp      |    32       |    8.34 sec/epoll     |  2.76 sec/epoll
     mnist_cnn      |    128      |    3.24 sec/epoll     |  1.29 sec/epoll
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


  Tensor(const vector<int> &shape, bool random = false, T range = 0) {
    size_t len = setup_tensor(shape);

    if (!random)
      return;

    auto random_uniform = [&](int size) {
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


class Softmax {

public:
  Softmax() {
  }

  ~Softmax() {
  }

  Tensor<float> forward(const Tensor<float> &x) const {
    float alpha = 1.0f, beta = 0.0f;

    Tensor<float> ans(x.shape);

    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, ans.dataTensor->get(), (float*)ans.d_data->get()));
    return ans;
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) const {
    if (lastLayer)
      return dy;

    Tensor<float> dx(x.shape, 0.0f);
    // float alpha = -float(count() / this->shape[0]) / this->shape[0], beta = 0.0f;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
      &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
      &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return dx;
  }

  Tensor<float> loss(const Tensor<float> &x, const Tensor<float> &y_real) const {
    assert(x.shape == y_real.shape);

    float posi = 1.0f / x.shape[0], nega = -1.0f / x.shape[0], batch = 1.0f / x.shape[0];
    size_t len = x.count();

    // Tensor<float> loss = y_real;
    // assert(CUDNN_STATUS_SUCCESS == cudnnAddTensor(cudnnHandle,
    //   &posi, this->dataTensor->get(), (float*)this->d_data->get(), &nega, loss.dataTensor->get(), (float*)loss.d_data->get()));

    Tensor<float> loss(x.shape, 0.0f);
    cublasSaxpy(cublasHandle, len, &posi, (float*)x.d_data->get(), 1, (float*)loss.d_data->get(), 1);
    cublasSaxpy(cublasHandle, len, &nega, (float*)y_real.d_data->get(), 1, (float*)loss.d_data->get(), 1);
    return loss;
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


class LRN {
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

  Tensor<float> forward(const Tensor<float> &x) const {
    assert(x.shape.size() == 4);

    Tensor<float> y(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelForward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));
    return move(y);
  }

  Tensor<float> backward(const Tensor<float> &dy, const Tensor<float> &y, const Tensor<float> &x, bool lastLayer = false) const {
    Tensor<float> dx(x.shape);
    if (lastLayer)
      return dx;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelBackward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
        x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }
};

class Pooling {
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

  Tensor<float> forward(const Tensor<float> &x) const {
    assert(x.shape.size() == 4);
    // assert((x.shape[2] - (size - stride)) % stride == 0 && (x.shape[3] - (size - stride)) % stride == 0);

    Tensor<float> ans({x.shape[0], x.shape[1], (x.shape[2] - (size - stride)) / stride, (x.shape[3] - (size - stride)) / stride});
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

  void learn(const vector<Tensor<float> > &tensors, float lr = -0.01) const {
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


int main() {
  Tensor<float>::init();

  Softmax softmax;
  Flatten flatten;
  Activation relu(CUDNN_ACTIVATION_RELU);
  Pooling pooling(2, 2, CUDNN_POOLING_MAX);

  // MNIST_MLP
  Dense mnist_fc1(512), mnist_fc2(512), mnist_fc3(10);

  // MNIST_CNN
  Dense mnist_dense1(128), mnist_dense2(10);
  Convolution mnist_cnn1(32, 3), mnist_cnn2(64, 3);

  // CIFAR10_LENET
  Dense lenet_dense1(512), lenet_dense2(10);
  Convolution lenet_cnn1(32, 5, true), lenet_cnn2(64, 5, true);

  // CIFAR10_ALEXNET
  Pooling alex_pooling(3, 2, CUDNN_POOLING_MAX);
  LRN alex_lrn(4, 1.0, 0.001 / 9.0, 0.75);
  Dense alex_fc1(384), alex_fc2(192), alex_fc3(10);
  Convolution alex_cnn1(64, 5, true), alex_cnn2(64, 5, true);

  auto full_images = ReadNormalDataset(TRAIN_IMAGES);
  auto full_labels = ReadNormalDataset(TRAIN_LABELS);
  assert(full_images.first[0] == full_labels.first[0]);

  int samples = full_images.first[0];
  int width = full_images.first[1] * full_images.first[2] * full_images.first[3];
  int classes = full_labels.first[1];
  printf("Total %d samples (%d, %d, %d) for %d classes found.\n", samples, full_images.first[1], full_images.first[2], full_images.first[3], classes);

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

    // << CIFAR10_ALEXNET >>
    auto y0 = images;
    auto y1 = alex_cnn1     .forward(y0); // 64x28x28
    auto y2 = relu          .forward(y1); // 64x28x28
    auto y3 = alex_pooling  .forward(y2); // 64x13x13
    auto y4 = alex_lrn      .forward(y3); // 64x13x13
    auto y5 = alex_cnn2     .forward(y4); // 64x9x9
    auto y6 = relu          .forward(y5); // 64x28x28
    auto y7 = alex_lrn      .forward(y6); // 64x9x9
    auto y8 = alex_pooling  .forward(y7); // 64x4x4
    auto y9 = flatten       .forward(y8); // 1024
    auto y10 = alex_fc1     .forward(y9); // 384
    auto y11 = relu         .forward(y10); // 64x28x28
    auto y12 = alex_fc2     .forward(y11); // 192
    auto y13 = relu         .forward(y12); // 64x28x28
    auto y14 = alex_fc3     .forward(y13); // 10
    auto y15 = softmax      .forward(y14); // 10

    auto dy14 = softmax.loss(y15, labels);
    auto dy13pack = alex_fc3.backward(dy14, y14, y13); alex_fc3.learn(dy13pack, lr); auto dy13 = dy13pack.back();
    auto dy12 = relu.backward(dy13, y13, y12);
    auto dy11pack = alex_fc2.backward(dy12, y12, y11); alex_fc2.learn(dy11pack, lr); auto dy11 = dy11pack.back();
    auto dy10 = relu.backward(dy11, y11, y10);
    auto dy9pack = alex_fc1.backward(dy10, y10, y9); alex_fc1.learn(dy9pack, lr); auto dy9 = dy9pack.back();
    auto dy8 = flatten.backward(dy9, y9, y8);
    auto dy7 = alex_pooling.backward(dy8, y8, y7);
    auto dy6 = alex_lrn.backward(dy7, y7, y6);
    auto dy5 = relu.backward(dy6, y6, y5);
    auto dy4pack = alex_cnn2.backward(dy5, y5, y4); alex_cnn2.learn(dy4pack, lr); auto dy4 = dy4pack.back();
    auto dy3 = alex_lrn.backward(dy4, y4, y3);
    auto dy2 = alex_pooling.backward(dy3, y3, y2);
    auto dy1 = relu.backward(dy2, y2, y1);
    auto dy0pack = alex_cnn1.backward(dy1, y1, y0, true); alex_cnn1.learn(dy0pack, lr); // auto dy0 = dy0pack.back();

    auto data_output = y14, data_loss = y15;


    /* << MNIST_LENET >>
    auto y0 = images;
    auto y1 = lenet_cnn1.forward(y0);
    auto y2 = pooling.forward(y1);
    auto y3 = lenet_cnn2.forward(y2);
    auto y4 = pooling.forward(y3);
    auto y5 = flatten.forward(y4);
    auto y6 = lenet_dense1.forward(y5);
    auto y7 = relu.forward(y6);
    auto y8 = lenet_dense2.forward(y7);
    auto y9 = softmax.forward(y8);

    auto dy8 = softmax.loss(y9, labels);
    auto dy7pack = lenet_dense2.backward(dy8, y8, y7); lenet_dense2.learn(dy7pack, lr); auto dy7 = dy7pack.back();
    auto dy6 = relu.backward(dy7, y7, y6);
    auto dy5pack = lenet_dense1.backward(dy6, y6, y5); lenet_dense1.learn(dy5pack, lr); auto dy5 = dy5pack.back();
    auto dy4 = flatten.backward(dy5, y5, y4);
    auto dy3 = pooling.backward(dy4, y4, y3);
    auto dy2pack = lenet_cnn2.backward(dy3, y3, y2); lenet_cnn2.learn(dy2pack, lr); auto dy2 = dy2pack.back();
    auto dy1 = pooling.backward(dy2, y2, y1);
    auto dy0pack = lenet_cnn1.backward(dy1, y1, y0, true); lenet_cnn1.learn(dy0pack, lr); // auto dy0 = dy0pack.back();

    auto data_output = y8, data_loss = y9;
    */

    /* << MNIST_CNN >>
    auto y0 = images;
    auto y1 = mnist_cnn1.forward(y0);
    auto y2 = relu.forward(y1);
    auto y3 = mnist_cnn2.forward(y2);
    auto y4 = relu.forward(y3);
    auto y5 = pooling.forward(y4);
    auto y6 = flatten.forward(y5);
    auto y7 = mnist_dense1.forward(y6);
    auto y8 = relu.forward(y7);
    auto y9 = mnist_dense2.forward(y8);
    auto y10 = softmax.forward(y9);

    auto dy9 = softmax.loss(y10, labels);
    auto dy8pack = mnist_dense2.backward(dy9, y9, y8); mnist_dense2.learn(dy8pack, lr); auto dy8 = dy8pack.back();
    auto dy7 = relu.backward(dy8, y8, y7);
    auto dy6pack = mnist_dense1.backward(dy7, y7, y6); mnist_dense1.learn(dy6pack, lr); auto dy6 = dy6pack.back();
    auto dy5 = flatten.backward(dy6, y6, y5);
    auto dy4 = pooling.backward(dy5, y5, y4);
    auto dy3 = relu.backward(dy4, y4, y3);
    auto dy2pack = mnist_cnn2.backward(dy3, y3, y2); mnist_cnn2.learn(dy2pack, lr); auto dy2 = dy2pack.back();
    auto dy1 = relu.backward(dy2, y2, y1);
    auto dy0pack = mnist_cnn1.backward(dy1, y1, y0, true); mnist_cnn1.learn(dy0pack, lr); // auto dy0 = dy0pack.back();

    auto data_output = y9, data_loss = y10;
    */

    /* << MNIST_MLP >>
    auto y0 = flatten.forward(images);
    auto y1 = mnist_fc1.forward(y0);
    auto y2 = relu.forward(y1);
    auto y3 = mnist_fc2.forward(y2);
    auto y4 = relu.forward(y3);
    auto y5 = mnist_fc3.forward(y4);
    auto y6 = softmax.forward(y5);

    auto dy5 = softmax.loss(y6, labels);
    auto dy4pack = mnist_fc3.backward(dy5, y4, y4); mnist_fc3.learn(dy4pack, lr); auto dy4 = dy4pack.back();
    auto dy3 = relu.backward(dy4, y4, y3);
    auto dy2pack = mnist_fc2.backward(dy3, y3, y2); mnist_fc2.learn(dy2pack, lr); auto dy2 = dy2pack.back();
    auto dy1 = relu.backward(dy2, y2, y1);
    auto dy0pack = mnist_fc1.backward(dy1, y1, y0); mnist_fc1.learn(dy0pack, lr); auto dy0 = dy0pack.back();

    auto data_output = y5, data_loss = y6;
    */

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

