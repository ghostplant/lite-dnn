/*
  mnist_mlp based on CUBLAS/CUDNN
  g++ -O3 -std=c++14 "$@" -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -lcblas -lcudnn

  Maintainer: Wei CUI <ghostplant@qq.com>

  Benchmark on Nvida Tesla P100:
  * CPP + CUDNN      mnist_mlp: 2.76 sec/epoll (batch_size = 32)
  * Keras + TF4CUDA  mnist_mlp: 8.34 sec/epoll (batch_size = 32)
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

class TensorHandler {
  cudnnTensorDescriptor_t dataTensor;

public:
  TensorHandler(const vector<int> &shape) {
    assert(CUDNN_STATUS_SUCCESS == cudnnCreateTensorDescriptor(&dataTensor));
    assert(shape.size() <= 4);
    int dimA[4] = {1, 1, 1, 1};
    for (int i = 0; i < shape.size(); ++i)
      dimA[i] = shape[i];
    assert(CUDNN_STATUS_SUCCESS == cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dimA[0], dimA[1], dimA[2], dimA[3]));
  }

  ~TensorHandler() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));
  }

  cudnnTensorDescriptor_t get() const {
    return dataTensor;
  }
};


static cudnnHandle_t cudnnHandle;
static cublasHandle_t cublasHandle;
static CUstream hStream = NULL;


template <class T> class Tensor {
  vector<int> shape;
  shared_ptr<DeviceMemory> d_data;
  shared_ptr<TensorHandler> dataTensor;

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
    dataTensor = make_shared<TensorHandler>(shape);

    if (host.size() > 0)
      assert(CUDA_SUCCESS == cuMemcpyHtoDAsync_v2((CUdeviceptr)d_data->get(), host.data(), len * sizeof(T), hStream));
  }

  Tensor(const vector<int> &shape, const T val): shape(shape) {
    size_t len = count();
    d_data = make_shared<DeviceMemory>(len * sizeof(T));
    dataTensor = make_shared<TensorHandler>(shape);

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
    mat.dataTensor = make_shared<TensorHandler>(shape);
    assert(mat.count() == count());
    return mat;
  }

  Tensor matmul(const Tensor<T> &that, bool transposeThis = false, bool transposeThat = false) const {
    // ans = this * &that;
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

  Tensor incbias(const Tensor<T> &bias, const Tensor<T> &ones) const {
    // self += ones * bias 'T;
    // ~ this.shape = (batch_size, chan); bias.shape = (1, chan); ones.shape == (batch_size, 1);
    assert(this->shape.size() == 2 && bias.shape.size() == 2 && this->shape[1] == bias.shape[1] && bias.shape[0] == 1);
    // return this->matadd(ones.matmul(bias));

    float alpha = 1.0f;
    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                bias.shape[1], this->shape[0], 1,
                &alpha,
                (T*)bias.d_data->get(), bias.shape[1], // B
                (T*)ones.d_data->get(), 1,  // 1
                &alpha,
                (T*)this->d_data->get(), this->shape[1]);   // self
    return *this;
  }

  void learn(const Tensor<T> &tensor, T lr = -0.01) const {
    assert(this->shape == tensor.shape);

    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &lr, (T*)tensor.d_data->get(), 1, (T*)this->d_data->get(), 1));
  }

  Tensor activationForward(cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID) const {
    float alpha = 1.0f, beta = 0.0f;

    Tensor<T> ans(this->shape);

    cudnnActivationDescriptor_t dataActivation;
    cudnnCreateActivationDescriptor(&dataActivation);
    assert(CUDNN_STATUS_SUCCESS == cudnnSetActivationDescriptor(dataActivation, mode, CUDNN_PROPAGATE_NAN, 0.0));
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationForward(cudnnHandle, dataActivation, &alpha, dataTensor->get(), (T*)this->d_data->get(), &beta, dataTensor->get(), (T*)ans.d_data->get()));
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyActivationDescriptor(dataActivation));
    return ans;
  }

  Tensor activationBackward(const Tensor<T> &tensorPost, const Tensor<T> &tensorPre, cudnnActivationMode_t mode) const {
    float alpha = 1.0f, beta = 0.0f;

    cudnnActivationDescriptor_t dataActivation;
    cudnnCreateActivationDescriptor(&dataActivation);
    assert(CUDNN_STATUS_SUCCESS == cudnnSetActivationDescriptor(dataActivation, mode, CUDNN_PROPAGATE_NAN, 0.0));
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationBackward(cudnnHandle, dataActivation, &alpha, dataTensor->get(), (T*)tensorPost.d_data->get(),
        dataTensor->get(), (T*)this->d_data->get(), dataTensor->get(), (T*)tensorPre.d_data->get(), &beta, dataTensor->get(), (T*)this->d_data->get()));
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyActivationDescriptor(dataActivation));
    return *this;
  }

  Tensor softmaxLoss(Tensor<T> &outputs) const {
    assert(this->shape.size() == 2 && this->shape == outputs.shape);

    Tensor<T> ans(this->shape, 0);

    float posi = 1.0f, nega = -1.0f, batch = 1.0 / this->shape[0];
    cublasSaxpy(cublasHandle, count(), &posi, (T*)this->d_data->get(), 1, (T*)ans.d_data->get(), 1);
    cublasSaxpy(cublasHandle, count(), &nega, (T*)outputs.d_data->get(), 1, (T*)ans.d_data->get(), 1);
    cublasSscal(cublasHandle, count(), &batch, (T*)ans.d_data->get(), 1);
    // float alpha = -float(count() / this->shape[0]) / this->shape[0], beta = 0.0f;
    // assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
    //   &alpha, dataTensor->get(), (T*)this->d_data->get(), dataTensor->get(), (T*)outputs.d_data->get(), &beta, dataTensor->get(), (T*)ans.d_data->get()));
    return ans;
  }

  Tensor softmaxForward() const {
    float alpha = 1.0f, beta = 0.0f;

    Tensor<T> ans(this->shape);

    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &alpha, dataTensor->get(), (T*)this->d_data->get(), &beta, dataTensor->get(), (T*)ans.d_data->get()));
    return ans;
  }

  Tensor poolingForward(int size, int stride, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX) const {
    assert(this->shape.size() == 4);

    float alpha = 1.0f, beta = 0.0f;

    Tensor<T> ans({this->shape[0], this->shape[1], this->shape[2] / stride, this->shape[3] / stride});

    cudnnPoolingDescriptor_t poolDesc;
    cudnnCreatePoolingDescriptor(&poolDesc);
    cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN, size, size, 0, 0, stride, stride);
    cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, dataTensor->get(), (T*)this->d_data->get(), &beta, ans.dataTensor->get(), (T*)ans.d_data->get());
    cudnnDestroyPoolingDescriptor(poolDesc);
    return ans;
  }

  Tensor poolingBackward(const Tensor<T> &tensorPost, const Tensor<T> &tensorPre, int size, int stride, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX) const {
    assert(this->shape.size() == 4);

    float alpha = 1.0f, beta = 0.0f;

    Tensor<T> ans({this->shape[0], this->shape[1], this->shape[2] * stride, this->shape[3] * stride});

    cudnnPoolingDescriptor_t poolDesc;
    cudnnCreatePoolingDescriptor(&poolDesc);
    cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN, size, size, 0, 0, stride, stride);
    cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, dataTensor->get(), (T*)tensorPost.d_data->get(), dataTensor->get(), (T*)this->d_data->get(),
                         ans.dataTensor->get(), (T*)tensorPre.d_data->get(), &beta, ans.dataTensor->get(), (T*)ans.d_data->get());
    cudnnDestroyPoolingDescriptor(poolDesc);
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

  Tensor convolutionForward(int filters, int kernel_h, int kernel_w) const {
    assert(this->shape.size() == 4);

    float alpha = 1.0f, beta = 0.0f;
    unsigned in_chan = this->shape[1], out_chan = this->shape[1] * filters;

    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_chan, in_chan, kernel_h, kernel_w);

    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int n, c, h, w;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolution2dForwardOutputDim(convDesc, this->dataTensor->get(), filterDesc, &n, &c, &h, &w));

    Tensor<T> ans({n, c, h, w});

    cudnnConvolutionFwdAlgo_t convalgo;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardAlgorithm(cudnnHandle, this->dataTensor->get(), filterDesc, convDesc,
        ans.dataTensor->get(), CUDNN_CONVOLUTION_FWD_NO_WORKSPACE /* CUDNN_CONVOLUTION_FWD_PREFER_FASTEST */, 0, &convalgo));

    size_t sizeInBytes = 0;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, this->dataTensor->get(), filterDesc, convDesc,
        ans.dataTensor->get(), convalgo, &sizeInBytes));
    assert(sizeInBytes == 0);

    Tensor<T> w_krnl({in_chan * out_chan * kernel_h * kernel_w}, 1.0f);
    assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionForward(cudnnHandle, &alpha, this->dataTensor->get(), (T*)this->d_data->get(), filterDesc,
        (T*)w_krnl.d_data->get(), convDesc, convalgo, NULL, sizeInBytes, &beta, ans.dataTensor->get(), (T*)ans.d_data->get()));

    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(filterDesc);

    // cudnnConvolutionBwdFilterAlgo_t convbwfalgo;
    // cudnnConvolutionBwdDataAlgo_t convbwdalgo;
    return ans;
  }

  /*
  Tensor batchNormalizationForward(cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_PER_ACTIVATION) const {
    assert(this->shape.size() == 4);

    float alpha = 1.0f, beta = 0.0f;

    Tensor<T> ans(this->shape);

    // mode == CUDNN_BATCHNORM_PER_ACTIVATION ==> 1xCxHxW
    // mode == CUDNN_BATCHNORM_SPATIAL ==> 1xCx1x1
    // assert(CUDNN_STATUS_SUCCESS == cudnnBatchNormalizationForwardInference(cudnnHandle, mode,
    //     &alpha, &beta, this->dataTensor->get(), (T*)this->d_data->get(), ans.dataTensor->get(), (T*)ans.d_data->get(),
    //     bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, CUDNN_BN_MIN_EPSILON));
    return ans;
  }

  cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardInference(
                                  cudnnHandle_t                       handle,
                                  cudnnBatchNormMode_t                mode,
                                  const void                         *alpha, // alpha[0] = result blend factor
                                  const void                         *beta,  // beta[0] = dest layer blend factor
                                  const cudnnTensorDescriptor_t       xDesc,
                                  const void                         *x,     // NxCxHxW
                                  const cudnnTensorDescriptor_t       yDesc,
                                  void                               *y,     // NxCxHxW
                                  const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,
                                  const void                         *bnScale,
                                  const void                         *bnBias,
                                  const void                         *estimatedMean,
                                  const void                         *estimatedVariance,
                                  double                              epsilon )

  cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationForwardTraining(
                                  cudnnHandle_t                       handle,
                                  cudnnBatchNormMode_t                mode,
                                  const void                         *alpha, // alpha[0] = result blend factor
                                  const void                         *beta,  // beta[0] = dest layer blend factor
                                  const cudnnTensorDescriptor_t       xDesc,
                                  const void                         *x,     // NxCxHxW
                                  const cudnnTensorDescriptor_t       yDesc,
                                  void                               *y,     // NxCxHxW
                                  const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,
                                  const void                         *bnScale,
                                  const void                         *bnBias,
                                  double                              exponentialAverageFactor,
                                  void                               *resultRunningMean,
                                  void                               *resultRunningVariance,
                                  double                              epsilon,
                                  void                               *resultSaveMean,
                                  void                               *resultSaveInvVariance )

  cudnnStatus_t CUDNNWINAPI cudnnBatchNormalizationBackward(
                                  cudnnHandle_t                       handle,
                                  cudnnBatchNormMode_t                mode,
                                  const void                         *alphaDataDiff,
                                  const void                         *betaDataDiff,
                                  const void                         *alphaParamDiff,
                                  const void                         *betaParamDiff,
                                  const cudnnTensorDescriptor_t       xDesc, // same desc for x, dx, dy
                                  const void                         *x,
                                  const cudnnTensorDescriptor_t       dyDesc,
                                  const void                         *dy,
                                  const cudnnTensorDescriptor_t       dxDesc,
                                  void                               *dx,
                                  const cudnnTensorDescriptor_t       dBnScaleBiasDesc,
                                  const void                         *bnScale, // bnBias doesn't affect backpropagation
                                  void                               *dBnScaleResult,
                                  void                               *dBnBiasResult,
                                  double                              epsilon,
                                  const void                         *savedMean,
                                  const void                         *savedInvVariance )
  */
};


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
    => input_shape = (batch_size, 784)
    + Layer::Dense(512)
    + Layer::Dense(512)
    + Layer::Dense(10)
    => output_shape: (batch_size, 10)
  */

  srand(10);

  auto random_uniform = [&](int size) {
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
  };

  Tensor<float>::init();

  int batch_size = 128;
  Tensor<float> ones({batch_size, 1}, 1),
      w_fc1({512, 784}, random_uniform(784 * 512)), w_fc1bias({1, 512}, random_uniform(512)),
      w_fc2({512, 512}, random_uniform(512 * 512)), w_fc2bias({1, 512}, random_uniform(512)),
      w_fc3({10, 512}, random_uniform(512 * 10)), w_fc3bias({1, 10}, random_uniform(10));

  vector<float> in(784 * batch_size), out(10 * batch_size);

  auto dataset = ReadUByteDataset(MNIST_IMAGES, MNIST_LABELS);
  int samples = dataset.first.size() / 784;
  printf("Total %d samples found.\n", samples);

  int it = 0, epochs = 100, steps = (samples + batch_size - 1) / batch_size * epochs;
  for (int k = 0; k < steps; ++k) {
    memset(out.data(), 0, out.size() * sizeof(float));
    for (int i = 0; i < batch_size; ++i, it = (it + 1) % samples) {
      int lb = dataset.second[it * 1];
      out[i * 10 + lb] = 1.0f;
      for (int j = 0; j < 784; ++j)
        in[i * 784 + j] = dataset.first[it * 784 + j] / 255.0;
    }

    Tensor<float> images({batch_size, 1, 28, 28}, in), labels({batch_size, 10}, out);

    images = images.reshape({batch_size, 784});

    auto fc1_out = images.matmul(w_fc1, false, true).incbias(w_fc1bias, ones);   // shape = {batch_size, 512}
    auto fc1_act = fc1_out.activationForward(CUDNN_ACTIVATION_RELU);              // shape = {batch_size, 512}

    auto fc2_out = fc1_act.matmul(w_fc2, false, true).incbias(w_fc2bias, ones);  // shape = {batch_size, 512}
    auto fc2_act = fc2_out.activationForward(CUDNN_ACTIVATION_RELU);              // shape = {batch_size, 512}

    auto fc3_out = fc2_act.matmul(w_fc3, false, true).incbias(w_fc3bias, ones);  // shape = {batch_size, 10}
    auto fc3_act = fc3_out.softmaxForward();                                      // shape = {batch_size, 10}

    auto fc3_dloss = fc3_act.softmaxLoss(labels);                                 // shape = {batch_size, 10}
    auto fc3_grad_w = fc3_dloss.matmul(fc2_act, true, false);                    // shape = {10, 512}
    auto fc3_grad_b = ones.matmul(fc3_dloss, true, false);                       // shape = {10, 1}

    auto fc2_dloss = fc3_dloss.matmul(w_fc3).activationBackward(fc2_act, fc2_out, CUDNN_ACTIVATION_RELU); // shape = {batch_size, 512}
    auto fc2_grad_w = fc2_dloss.matmul(fc1_act, true, false);                                             // shape = {512, 512}
    auto fc2_grad_b = ones.matmul(fc2_dloss, true, false);                                                // shape = {512, 1}

    auto fc1_dloss = fc2_dloss.matmul(w_fc2).activationBackward(fc1_act, fc1_out, CUDNN_ACTIVATION_RELU); // shape = {batch_size, 512}
    auto fc1_grad_w = fc1_dloss.matmul(images, true, false);                                              // shape = {512, 784}
    auto fc1_grad_b = ones.matmul(fc1_dloss, true, false);                                                // shape = {512, 1}

    w_fc1.learn(fc1_grad_w);
    w_fc1bias.learn(fc1_grad_b);
    w_fc2.learn(fc2_grad_w);
    w_fc2bias.learn(fc2_grad_b);
    w_fc3.learn(fc3_grad_w);
    w_fc3bias.learn(fc3_grad_b);


    if (it < batch_size) {
      vector<float> loss_data = fc3_dloss.get_data();
      float loss = 0.0f;
      for (int i = 0; i < loss_data.size(); ++i) {
        float j = fabs(loss_data[i]);
        if (j >= 1e-8)
          loss += -j * log(j);
      }
      vector<float> pred_data = fc3_act.get_data();
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
      static int step = 0;
      static unsigned long prev = 0LU;
      struct timeval tv;
      gettimeofday(&tv, NULL);
      unsigned long current = tv.tv_sec * 1000000LU + tv.tv_usec;

      printf("epoch = %d: loss = %.4f, acc = %.2f%%, time = %.4lfs\n", ++step, loss, acc * 100.0f / tot, prev ? (current - prev) * 1e-6 : 0);
      prev = current;
    }
  }
  return 0;
}

