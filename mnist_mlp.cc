/*
  mnist_mlp based on CUBLAS/CUDNN
  g++ -O3 -std=c++14 "$@" -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcblas -lcudnn

  Maintainer: Wei CUI <ghostplant@qq.com>

  Benchmark:
  (CPP + CUDNN)     mnist_mlp: 2.76 sec/epoll (batch_size = 32)
  (Keras + TF_CUDA) mnist_mlp: 8.34 sec/epoll (batch_size = 32)
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


class DeviceMemory {
  void *d_data;

public:
  DeviceMemory(size_t length) {
    assert(CUDA_SUCCESS == cuMemAlloc_v2((CUdeviceptr*)&d_data, length));
  }

  ~DeviceMemory() {
    assert(CUDA_SUCCESS == cuMemFree_v2((CUdeviceptr)d_data));
  }

  void* get() const {
    return d_data;
  }
};


static cudnnHandle_t cudnnHandle;
static cublasHandle_t cublasHandle;
static CUstream hStream = NULL;


template <class T> class Tensor {
  vector<int> shape;
  shared_ptr<DeviceMemory> d_data;

public:

  static void init() {
    int devCount = 0;
    CUcontext primaryCtx;

    cuInit(0);
    cuDeviceGetCount(&devCount);
    assert(devCount > 0);
    cuDevicePrimaryCtxRetain(&primaryCtx, 0);
    cuCtxSetCurrent(primaryCtx);

    cublasCreate(&cublasHandle);
    cudnnCreate(&cudnnHandle);
  }

  Tensor(const vector<int> &shape, const vector<T> &host = {}): shape(shape) {
    size_t len = count();
    d_data = make_shared<DeviceMemory>(len * sizeof(T));

    if (host.size() > 0)
      assert(CUDA_SUCCESS == cuMemcpyHtoDAsync_v2((CUdeviceptr)d_data->get(), host.data(), len * sizeof(T), hStream));
  }

  Tensor(const vector<int> &shape, const T val): shape(shape) {
    size_t len = count();
    d_data = make_shared<DeviceMemory>(len * sizeof(T));

    unsigned int ui = (unsigned int&)val;
    assert(CUDA_SUCCESS == cuMemsetD32Async((CUdeviceptr)d_data->get(), ui, len, hStream));
  }

  size_t count() const {
    size_t len = 1;
    for (auto it: shape)
      len *= it;
    return len;
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

  Tensor reshape(const vector<int> &shape) const {
    Tensor mat = *this;
    mat.shape = shape;
    assert(mat.count() == count());
    return mat;
  }

  Tensor matmul(const Tensor<T> &that, bool transposeThis = false, bool transposeThat = false) const {
    // ans = *this * that;
    assert(this->shape.size() == 2 && that.shape.size() == 2);

    int ax = this->shape[0], ay = this->shape[1];
    if (transposeThis)
      swap(ax, ay);
    int bx = that.shape[0], by = that.shape[1];
    if (transposeThat)
      swap(bx, by);
    assert(ay == bx);

    Tensor<T> ans({ax, by});

    float alpha = 1.0f, beta = 0.0f;
    assert(0 == cublasSgemm(cublasHandle,
                            transposeThis ? CUBLAS_OP_T : CUBLAS_OP_N, transposeThat ? CUBLAS_OP_T : CUBLAS_OP_N,
                            ax, by, ay,
                            &alpha,
                            (T*)this->d_data->get(), this->shape[0], // X
                            (T*)that.d_data->get(), that.shape[0],  // Y
                            &beta,
                            (T*)ans.d_data->get(), ans.shape[0]));   // Z
    return ans;
  }

  Tensor incbias(const Tensor<T> &bias, const Tensor<T> &ones) const {
    // self += B * ones'T; -- ones.shape == 1 x batch_size
    assert(this->shape.size() == 2 && bias.shape.size() == 1 && this->shape[0] == bias.shape[0]);

    float alpha = 1.0f;
    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                bias.shape[0], this->shape[1], 1,
                &alpha,
                (T*)bias.d_data->get(), bias.shape[0], // B
                (T*)ones.d_data->get(), 1,  // 1
                &alpha,
                (T*)this->d_data->get(), this->shape[0]);   // self
    return *this;
  }

  Tensor activationForward(cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID) const {
    float alpha = 1.0f, beta = 0.0f;

    cudnnTensorDescriptor_t dataTensor;
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateTensorDescriptor(&dataTensor));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->shape.back(), count() / this->shape.back(), 1, 1));

    Tensor<T> ans(this->shape);

    cudnnActivationDescriptor_t dataActivation;
    cudnnCreateActivationDescriptor(&dataActivation);
    assert(CUDNN_STATUS_SUCCESS == cudnnSetActivationDescriptor(dataActivation, mode, CUDNN_PROPAGATE_NAN, 0.0));
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationForward(cudnnHandle, dataActivation, &alpha, dataTensor, (T*)this->d_data->get(), &beta, dataTensor, (T*)ans.d_data->get()));
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyActivationDescriptor(dataActivation));

    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));
    return ans;
  }

  Tensor activationBackward(cudnnActivationMode_t mode, const Tensor<T> &tensorAct, const Tensor<T> &tensor) const {
    float alpha = 1.0f, beta = 0.0f;

    cudnnTensorDescriptor_t dataTensor;
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateTensorDescriptor(&dataTensor));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->shape.back(), count() / this->shape.back(), 1, 1));

    cudnnActivationDescriptor_t dataActivation;
    cudnnCreateActivationDescriptor(&dataActivation);
    assert(CUDNN_STATUS_SUCCESS == cudnnSetActivationDescriptor(dataActivation, mode, CUDNN_PROPAGATE_NAN, 0.0));
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationBackward(cudnnHandle, dataActivation,
        &alpha, dataTensor, (T*)tensorAct.d_data->get(), dataTensor, (T*)this->d_data->get(), dataTensor, (T*)tensor.d_data->get(), &beta, dataTensor, (T*)this->d_data->get()));
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyActivationDescriptor(dataActivation));

    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));
    return *this;
  }
  
  void learn(const Tensor<T> &tensor, T lr = -0.01) const {
    assert(count() == tensor.count());

    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &lr, (T*)tensor.d_data->get(), 1, (T*)this->d_data->get(), 1));
  }

  Tensor softmaxLoss(Tensor<T> &outputs) const {
    assert(this->shape.size() == 2 && this->shape == outputs.shape);

    cudnnTensorDescriptor_t dataTensor;
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateTensorDescriptor(&dataTensor));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->shape.back(), count() / this->shape.back(), 1, 1));

    Tensor<T> ans(this->shape, 0);

    float posi = 1.0f, nega = -1.0f, batch = 1.0 / this->shape.back();
    cublasSaxpy(cublasHandle, count(), &posi, (T*)this->d_data->get(), 1, (T*)ans.d_data->get(), 1);
    cublasSaxpy(cublasHandle, count(), &nega, (T*)outputs.d_data->get(), 1, (T*)ans.d_data->get(), 1);
    cublasSscal(cublasHandle, count(), &batch, (T*)ans.d_data->get(), 1);
    // float alpha = -float(count() / this->shape.back()) / this->shape.back(), beta = 0.0f;
    // assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    //   &alpha, dataTensor, (T*)this->d_data->get(), dataTensor, (T*)outputs.d_data->get(), &beta, dataTensor, (T*)ans.d_data->get()));
    // assert(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));

    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));
    return ans;
  }

  Tensor softmaxForward() const {
    float alpha = 1.0f, beta = 0.0f;

    cudnnTensorDescriptor_t dataTensor;
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateTensorDescriptor(&dataTensor));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, this->shape.back(), count() / this->shape.back(), 1, 1));

    Tensor<T> ans(this->shape);

    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha, dataTensor, (T*)this->d_data->get(), &beta, dataTensor, (T*)ans.d_data->get()));

    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));
    return ans;
  }
};


vector<float> random_uniform(int size) {
  vector<float> r(size);
  float avg1 = 0.0f, avg2 = 0.0f, dev;

  for (int i = 0; i < r.size(); ++i) {
    r[i] = rand() / float(RAND_MAX);
    avg1 += r[i], avg2 += r[i] * r[i];
  }
  avg1 /= r.size(), avg2 /= r.size(), dev = sqrt(avg2 - avg1 * avg1);

  for (int i = 0; i < r.size(); ++i)
    r[i] = (r[i] - avg1) / dev;
  return move(r);
}

pair<vector<uint8_t>, vector<uint8_t> > ReadUByteDataset(const char* image_filename, const char* label_filename) {
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

int main() {
  /*
    @input_shape = batch_size * 784
    + Layer::Dense(512)
    + Layer::Dense(512)
    + Layer::Dense(10)
    @output_shape: batch_size * 10
  */

  Tensor<float>::init();
  
  // srand(time(0));

  int batch_size = 128;
  Tensor<float> ones({batch_size, 1}, 1),
      w_fc1({784, 512}, random_uniform(784 * 512)), w_fc1bias({512}, random_uniform(512)),
      w_fc2({512, 512}, random_uniform(512 * 512)), w_fc2bias({512}, random_uniform(512)),
      w_fc3({512, 10}, random_uniform(512 * 10)), w_fc3bias({10}, random_uniform(10));

  vector<float> in(784 * batch_size), out(10 * batch_size);

  auto dataset = ReadUByteDataset(MNIST_IMAGES, MNIST_LABELS);
  int samples = dataset.first.size() / 784;
  printf("Total %d samples found.\n", samples);

  for (int t = 0, it = 0; t < 102400; ++t) {
    memset(out.data(), 0, out.size() * sizeof(float));
    for (int i = 0; i < batch_size; ++i, it = (it + 1) % samples) {
      int lb = dataset.second[it * 1];
      out[i * 10 + lb] = 1.0f;
      for (int j = 0; j < 784; ++j)
        in[i * 784 + j] = dataset.first[it * 784 + j] / 255.0;
    }

    Tensor<float> images({784, batch_size}, in), labels({10, batch_size}, out);

    auto fc1_out = w_fc1.matmul(images, true, false).incbias(w_fc1bias, ones);   // shape = {512, batch_size}
    auto fc1_act = fc1_out.activationForward(CUDNN_ACTIVATION_RELU);              // shape = {512, batch_size}

    auto fc2_out = w_fc2.matmul(fc1_act, true, false).incbias(w_fc2bias, ones);  // shape = {512, batch_size}
    auto fc2_act = fc2_out.activationForward(CUDNN_ACTIVATION_RELU);              // shape = {512, batch_size}

    auto fc3_out = w_fc3.matmul(fc2_act, true, false).incbias(w_fc3bias, ones);  // shape = {10, batch_size}
    auto fc3_act = fc3_out.softmaxForward();                                      // shape = {10, batch_size}

    auto fc3_dloss = fc3_act.softmaxLoss(labels);                                 // shape = {10, batch_size}
    auto fc3_grad_w = fc2_act.matmul(fc3_dloss, false, true);                    // shape = {512, 10}
    auto fc3_grad_b = fc3_dloss.matmul(ones);                                     // shape = {10, 1}
    auto fc2_dloss = w_fc3.matmul(fc3_dloss).activationBackward(CUDNN_ACTIVATION_RELU, fc2_act, fc2_out); // shape = {512, batch_size}

    auto fc2_grad_w = fc1_act.matmul(fc2_dloss, false, true);                                             // shape = {512, 512}
    auto fc2_grad_b = fc2_dloss.matmul(ones);                                                              // shape = {512, 1}
    auto fc1_dloss = w_fc2.matmul(fc2_dloss).activationBackward(CUDNN_ACTIVATION_RELU, fc1_act, fc1_out); // shape = {512, batch_size}

    auto fc1_grad_w = images.matmul(fc1_dloss, false, true);                                              // shape = {784, 512}
    auto fc1_grad_b = fc1_dloss.matmul(ones);                                                              // shape = {512, 1}

    if (it < batch_size) {
      vector<float> loss_data = fc3_dloss.get_data();
      float loss = 0.0f;
      for (int i = 0; i < loss_data.size(); ++i) {
        float j = fabs(loss_data[i]);
        if (j >= 1e-8)
          loss += -j * log(j);
      }
      static int step = 0;
      static unsigned long prev = 0LU;
      struct timeval tv;
      gettimeofday(&tv, NULL);
      unsigned long current = tv.tv_sec * 1000000LU + tv.tv_usec;
      printf("epoch = %d: loss = %.4f, time = %.4lf s\n", ++step, loss, prev ? (current - prev) * 1e-6 : 0);
      prev = current;
    }

    w_fc1.learn(fc1_grad_w);
    w_fc1bias.learn(fc1_grad_b);
    w_fc2.learn(fc2_grad_w);
    w_fc2bias.learn(fc2_grad_b);
    w_fc3.learn(fc3_grad_w);
    w_fc3bias.learn(fc3_grad_b);
  }
  return 0;
}

