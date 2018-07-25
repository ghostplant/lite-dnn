#ifndef __LITEDNN_TENSOR__
#define __LITEDNN_TENSOR__

#include <vector>
#include <memory>
#include <unordered_map>

using namespace std;

static unordered_map<size_t, vector<void*>> cached_mem;
static cudnnHandle_t cudnnHandle;
static cublasHandle_t cublasHandle;
static CUstream hStream = NULL;


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


class Tensor {

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
  }


  Tensor(): shape({0}) {
  }

  Tensor(const vector<int> &shape, bool random = false) {
    size_t len = setup_tensor(shape);

    if (!random)
      return;
    // glorot_normal
    float receptive = 0.5f;
    for (int i = 2; i < shape.size(); ++i)
      receptive *= shape[i];
    assert(shape.size() >= 2);
    float limit = sqrt(3.0f / max(1.0f, (shape[0] + shape[1]) * receptive));

    auto random_uniform = [&]() {
      // srand(shape.size());
      // devi = sqrt(avg2 - avg1 * avg1);

      vector<float> r(len);
      for (int i = 0; i < r.size(); ++i)
        r[i] = rand() * 2.0f * limit / RAND_MAX - limit;
      return move(r);
    };

    set_data(random_uniform().data());
  }

  Tensor(const vector<int> &shape, const float *host) {
    size_t len = setup_tensor(shape);

    set_data(host);
  }

  Tensor(const vector<int> &shape, const float val) {
    size_t len = setup_tensor(shape);

    assert(sizeof(float) == sizeof(unsigned int));
    unsigned int ui = (unsigned int&)val;
    assert(CUDA_SUCCESS == cuMemsetD32Async((CUdeviceptr)d_data->get(), ui, len, hStream));
  }


  size_t setup_tensor(const vector<int> &shape) {
    this->shape = shape;
    size_t len = count();
    d_data = make_shared<DeviceMemory>(len * sizeof(float));
    dataTensor = make_shared<TensorHandler>(shape);
    return len;
  }

  size_t count() const {
    size_t len = 1;
    for (auto it: shape)
      len *= it;
    return len;
  }

  void set_data(const float *host) const {
    size_t len = count();
    assert(CUDA_SUCCESS == cuMemcpyHtoDAsync_v2((CUdeviceptr)d_data->get(), host, len * sizeof(float), hStream));
    assert(CUDA_SUCCESS == cuStreamSynchronize(hStream));
  }

  vector<float> get_data() const {
    size_t len = count();
    vector<float> host(len);
    assert(CUDA_SUCCESS == cuMemcpyDtoHAsync_v2(host.data(), (CUdeviceptr)d_data->get(), len * sizeof(float), hStream));
    assert(CUDA_SUCCESS == cuStreamSynchronize(hStream));
    return move(host);
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
    Tensor ans(this->shape);
    assert(CUDA_SUCCESS == cuMemcpyDtoDAsync_v2((CUdeviceptr)ans.d_data->get(), (CUdeviceptr)this->d_data->get(), ans.count() * sizeof(float), hStream));
    return move(ans);
  }

  Tensor matmul(const Tensor &that, bool transposeThis = false, bool transposeThat = false) const {
    // ans = &that * this;
    const Tensor *A = &that, *B = this;
    bool transposeA = transposeThat, transposeB = transposeThis;

    assert(A->shape.size() == 2 && B->shape.size() == 2);

    int ax = A->shape[1], ay = A->shape[0];
    if (transposeA)
      swap(ax, ay);
    int bx = B->shape[1], by = B->shape[0];
    if (transposeB)
      swap(bx, by);
    assert(ay == bx);

    Tensor ans({by, ax});

    float alpha = 1.0f, beta = 0.0f;
    assert(0 == cublasSgemm(cublasHandle,
                            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                            ax, by, ay, &alpha,
                            (float*)A->d_data->get(), A->shape[1], // X
                            (float*)B->d_data->get(), B->shape[1],  // Y
                            &beta, (float*)ans.d_data->get(), ans.shape[1]));   // Z
    return ans;
  }

  Tensor add(const Tensor &that) const {
    assert(this->shape == that.shape);
    float alpha = 1.0f;
    Tensor ans(this->shape, 0.0f);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &alpha, (float*)this->d_data->get(), 1, (float*)ans.d_data->get(), 1));
    // Tensor ans = this->copy();
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &alpha, (float*)that.d_data->get(), 1, (float*)ans.d_data->get(), 1));
    return ans;
  }
};


pair<float, float> get_loss_and_accuracy(const Tensor &data_pred, const Tensor &data_label) {
  assert(data_pred.shape.size() == 2 && data_pred.shape == data_label.shape);

  vector<float> pred_data = data_pred.get_data();
  vector<float> real_data = data_label.get_data();

  float loss = 0.0f;
  for (int i = 0; i < pred_data.size(); ++i) {
    if (fabs(real_data[i]) >= 1e-8)
      loss += -real_data[i] * log(pred_data[i]);
  }
  loss /= pred_data.size();

  int tot = 0, acc = 0;
  for (int i = 0; i < data_pred.shape[0]; ++i) {
    int it = 0, jt = 0;
    for (int j = 1; j < data_pred.shape[1]; ++j) {
      if (pred_data[i * data_pred.shape[1] + it] < pred_data[i * data_pred.shape[1] + j])
        it = j;
      if (real_data[i * data_pred.shape[1] + jt] < real_data[i * data_pred.shape[1] + j])
        jt = j;
    }
    ++tot;
    if (it == jt)
      ++acc;
  }
  return {loss, acc * 100.0f / tot};
}

#endif
