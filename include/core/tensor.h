#ifndef __LITEDNN_TENSOR__
#define __LITEDNN_TENSOR__

#include <vector>
#include <memory>
#include <unordered_map>
#include <random>
#include <queue>
#include <algorithm>


#ifdef assert
#undef assert
#endif

#define die_if(__cond__, __desc__, ...) ({if (__cond__) { printf("  \033[33m[!] <<file %s:%d>> " __desc__ "\033[0m\n\n", __FILE__, __LINE__, ##__VA_ARGS__); fflush(stdout); exit(1);}})
#define assert(__cond__)  die_if(!(__cond__), "Assertion failed: %s.", #__cond__)

using namespace std;

struct DeviceResources {
  CUstream hStream;
  CUcontext hContext;
  cudnnHandle_t hCudnn;
  cublasHandle_t hCublas;
};

static vector<DeviceResources> devices;

static vector<unordered_map<size_t, vector<void*>>> cached_mem;
static cudnnHandle_t cudnnHandle;
static cublasHandle_t cublasHandle;
static int currentDev;


class DeviceMemory {
  void *d_data;
  size_t length;

public:
  DeviceMemory(size_t length): d_data(NULL), length(length) {
    if (length) {
      if (cached_mem.size() < devices.size()) cached_mem.resize(devices.size());
      auto& it = cached_mem[currentDev][length]; if (it.size()) { d_data = it.back(); it.pop_back(); return; }
      die_if(CUDA_SUCCESS != cuMemAlloc_v2((CUdeviceptr*)&d_data, length), "No more memory to allocate new buffer of size %zd B.", length);
    }
  }

  ~DeviceMemory() {
    if (d_data) {
      cached_mem[currentDev][length].push_back(d_data); return;
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
  bool trainable;

  static int deviceCount() {
    return devices.size();
  }

  static void synchronizeCurrentDevice() {
    assert(CUDA_SUCCESS == cuStreamSynchronize(devices[currentDev].hStream));
  }
  
  static void activateCurrentDevice(int dev) {
    assert(dev < devices.size());
    currentDev = dev;
    assert(CUDA_SUCCESS == cuCtxSetCurrent(devices[dev].hContext));
    cublasHandle = devices[dev].hCublas;
    cudnnHandle = devices[dev].hCudnn;
    assert(CUBLAS_STATUS_SUCCESS == cublasSetStream_v2(cublasHandle, devices[currentDev].hStream));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetStream(cudnnHandle, devices[currentDev].hStream));
  }

  static void init(bool randTime = false) {
    int devCount = 0;
    CUcontext primaryCtx;

    if (randTime)
      srand(time(0));

    assert(CUDA_SUCCESS == cuInit(0));
    assert(CUDA_SUCCESS == cuDeviceGetCount(&devCount));
    die_if(devCount <= 0, "No available GPUs detected.");

    devices.resize(devCount);

    for (int i = 0; i < devCount; ++i) {
      assert(CUDA_SUCCESS == cuDevicePrimaryCtxRetain(&devices[i].hContext, i));
      assert(CUDA_SUCCESS == cuCtxSetCurrent(devices[i].hContext));
      assert(CUDA_SUCCESS == cuStreamCreate(&devices[i].hStream, CU_STREAM_NON_BLOCKING));
      assert(CUBLAS_STATUS_SUCCESS == cublasCreate(&devices[i].hCublas));
      assert(CUBLAS_STATUS_SUCCESS == cublasSetPointerMode_v2(devices[i].hCublas, CUBLAS_POINTER_MODE_HOST));
      assert(CUDNN_STATUS_SUCCESS == cudnnCreate(&devices[i].hCudnn));
    }

    activateCurrentDevice(0);
  }

  static string stringify_shape(const vector<int> &shape, int offset = 0) {
    string ans = "(";
    if (offset == shape.size())
      return ans + ")";
    for (int i = offset; i < shape.size(); ++i)
      ans += to_string(shape[i]) + ((i + 1 < shape.size()) ? ", " : ")");
    return ans;
  }


  Tensor(): shape({0}) {
  }

  Tensor(const vector<int> &shape, bool random = false) {
    size_t len = setup_tensor(shape);

    if (!random)
      return;
    /* glorot_normal
    float receptive = 0.5f;
    for (int i = 2; i < shape.size(); ++i)
      receptive *= shape[i];
    assert(shape.size() >= 2);
    float limit = sqrt(3.0f / max(1.0f, (shape[0] + shape[1]) * receptive));
    r[i] = rand() * 2.0f * limit / RAND_MAX - limit;
    */

    float feed = 1.0f;
    for (int i = 1; i < shape.size(); ++i)
      feed *= shape[i];
    feed = sqrt(2.0f / feed);

    auto random_uniform = [&]() {
      std::default_random_engine generator;
      std::normal_distribution<float> normal(0.0f, 1.0f);

      vector<float> r(len);
      for (int i = 0; i < r.size(); ++i)
        r[i] = normal(generator) * feed;
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
    assert(CUDA_SUCCESS == cuMemsetD32Async((CUdeviceptr)d_data->get(), ui, len, devices[currentDev].hStream));
  }


  size_t setup_tensor(const vector<int> &shape) {
    this->shape = shape;
    this->trainable = true;
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
    assert(CUDA_SUCCESS == cuMemcpyHtoDAsync_v2((CUdeviceptr)d_data->get(), host, len * sizeof(float), devices[currentDev].hStream));
    assert(CUDA_SUCCESS == cuStreamSynchronize(devices[currentDev].hStream));
  }

  vector<float> get_data() const {
    size_t len = count();
    vector<float> host(len);
    if (len > 0) {
      assert(CUDA_SUCCESS == cuMemcpyDtoHAsync_v2(host.data(), (CUdeviceptr)d_data->get(), len * sizeof(float), devices[currentDev].hStream));
      assert(CUDA_SUCCESS == cuStreamSynchronize(devices[currentDev].hStream));
    }
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
    assert(CUDA_SUCCESS == cuMemcpyDtoDAsync_v2((CUdeviceptr)ans.d_data->get(), (CUdeviceptr)this->d_data->get(), ans.count() * sizeof(float), devices[currentDev].hStream));
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
                            (float*)A->d_data->get(), A->shape[1],  // X
                            (float*)B->d_data->get(), B->shape[1],  // Y
                            &beta, (float*)ans.d_data->get(), ans.shape[1]));  // Z
    return ans;
  }

  float energy() {
    double ans = 0.0;
    auto d = this->get_data();
    for (auto it: d)
      ans += it * it;
    return ans;
  }

  Tensor self_update(const Tensor &that, float alpha = 1.0f, float beta = 0.0f) const {
    assert(this->shape == that.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnTransformTensor(cudnnHandle,
        &alpha, that.dataTensor->get(), (float*)that.d_data->get(),
        &beta, this->dataTensor->get(), (float*)this->d_data->get()));
    return *this;
  }

  Tensor self_add(const Tensor &that, float ceof = 1.0f) const {
    assert(this->shape == that.shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, that.count(), &ceof, (float*)that.d_data->get(), 1, (float*)this->d_data->get(), 1));
    return *this;
  }

  Tensor add(const Tensor &that, float ceof = 1.0f) const {
    assert(this->shape == that.shape);
    Tensor ans(this->shape, 0.0f);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &ceof, (float*)this->d_data->get(), 1, (float*)ans.d_data->get(), 1));
    // Tensor ans = this->copy();
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &ceof, (float*)that.d_data->get(), 1, (float*)ans.d_data->get(), 1));
    return ans;
  }

  Tensor clip_by_value(float min_value, float max_value) const {
    Tensor left(this->shape, min_value);
    Tensor right(this->shape, max_value);
    Tensor interm(this->shape);

    float alpha = 1.0f, beta = 0.0f;
    cudnnOpTensorDescriptor_t op_desc;
    cudnnCreateOpTensorDescriptor(&op_desc);

    cudnnSetOpTensorDescriptor(op_desc, CUDNN_OP_TENSOR_MAX, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
    assert(CUDNN_STATUS_SUCCESS == cudnnOpTensor(cudnnHandle, op_desc,
      &alpha, this->dataTensor->get(), (float*)left.d_data->get(),
      &alpha, this->dataTensor->get(), (float*)this->d_data->get(),
      &beta, this->dataTensor->get(), (float*)interm.d_data->get()));

    cudnnSetOpTensorDescriptor(op_desc, CUDNN_OP_TENSOR_MIN, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
    assert(CUDNN_STATUS_SUCCESS == cudnnOpTensor(cudnnHandle, op_desc,
      &alpha, this->dataTensor->get(), (float*)right.d_data->get(),
      &alpha, this->dataTensor->get(), (float*)interm.d_data->get(),
      &beta, this->dataTensor->get(), (float*)left.d_data->get()));
    cudnnDestroyOpTensorDescriptor(op_desc);
    return left;
  }

  pair<float, float> get_loss_and_accuracy_with(const Tensor &data_label) {
    const Tensor &data_pred = *this;
    assert(data_pred.shape.size() == 2 && data_pred.shape == data_label.shape);

    vector<float> pred_data = data_pred.clip_by_value(1.0e-7f, 1.0f - 1.0e-7f).get_data();
    vector<float> real_data = data_label.get_data();

    float loss = 0.0f;
    for (int i = 0; i < pred_data.size(); ++i) {
      loss -= real_data[i] * log(pred_data[i]) + (1.0f - real_data[i]) * log(1.0f - pred_data[i]);
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
};

#endif
