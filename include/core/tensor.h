#ifndef __LITEDNN_TENSOR__
#define __LITEDNN_TENSOR__

#include <vector>
#include <memory>
#include <unordered_map>
// #include <random>
#include <queue>
#include <algorithm>


#define die_if(__cond__, __desc__, ...) ({if (__cond__) { printf("  \033[33m[!] <<file %s:%d>> " __desc__ "\033[0m\n\n", __FILE__, __LINE__, ##__VA_ARGS__); fflush(stdout); Tensor::quit(1);}})
#define ensure(__cond__)  die_if(!(__cond__), "Condition checking failed: %s.", #__cond__)

using namespace std;


static int inline u_rand(unsigned int *seed) {
  *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
  return *seed;
}

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
static int currentDev = -1, globalStop;
static volatile int activeThread = 0;


class Tensor {

public:
  // Global Tensor Funtions

  static void init() {
    int devCount = 0;

    ensure(CUDA_SUCCESS == cuInit(0));
    ensure(CUDA_SUCCESS == cuDeviceGetCount(&devCount));
    die_if(devCount <= 0, "No available GPUs detected.");

    devices.resize(devCount);

    for (int i = 0; i < devCount; ++i) {
      ensure(CUDA_SUCCESS == cuDevicePrimaryCtxRetain(&devices[i].hContext, i));
      ensure(CUDA_SUCCESS == cuCtxSetCurrent(devices[i].hContext));
      ensure(CUDA_SUCCESS == cuStreamCreate(&devices[i].hStream, CU_STREAM_NON_BLOCKING));
      ensure(CUBLAS_STATUS_SUCCESS == cublasCreate(&devices[i].hCublas));
      ensure(CUBLAS_STATUS_SUCCESS == cublasSetPointerMode_v2(devices[i].hCublas, CUBLAS_POINTER_MODE_HOST));
      ensure(CUDNN_STATUS_SUCCESS == cudnnCreate(&devices[i].hCudnn));
    }

    activateCurrentDevice(0);
  }

  static void quit(int exitCode = 0) {
    globalStop = true;
    while (activeThread)
      usleep(50000);
    exit(exitCode);
  }
  static int deviceCount() {
    return devices.size();
  }

  static void synchronizeCurrentDevice() {
    ensure(CUDA_SUCCESS == cuStreamSynchronize(devices[currentDev].hStream));
  }
  
  static void activateCurrentDevice(int dev) {
    ensure(dev < devices.size());
    if (currentDev == dev)
      return;
    currentDev = dev;
    ensure(CUDA_SUCCESS == cuCtxSetCurrent(devices[dev].hContext));
    cublasHandle = devices[dev].hCublas;
    cudnnHandle = devices[dev].hCudnn;
    ensure(CUBLAS_STATUS_SUCCESS == cublasSetStream_v2(cublasHandle, devices[currentDev].hStream));
    ensure(CUDNN_STATUS_SUCCESS == cudnnSetStream(cudnnHandle, devices[currentDev].hStream));
  }


  // DeviceMemory for Tensor

  class DeviceMemory {
    void *d_data;
    size_t length;

  public:
    DeviceMemory(size_t length): d_data(NULL), length(length) {
      if (length) {
        if (cached_mem.size() < devices.size()) cached_mem.resize(devices.size());
        auto& it = cached_mem[currentDev][length]; if (it.size()) { d_data = it.back(); it.pop_back(); return; }
        die_if(cuMemAlloc_v2((CUdeviceptr*)&d_data, length) != CUDA_SUCCESS, "No more memory to allocate new buffer of size %zd B.", length);
      }
    }

    ~DeviceMemory() {
      if (d_data) {
        cached_mem[currentDev][length].push_back(d_data); return;
        die_if(cuMemFree_v2((CUdeviceptr)d_data) != CUDA_SUCCESS, "Failed to free memory buffer: %p.", d_data);
      }
    }

    void* get() const {
      return d_data;
    }
  };


  // TensorHandler for Tensor

  class TensorHandler {
    cudnnTensorDescriptor_t dataTensor;

  public:
    TensorHandler(const vector<int> &shape) {
      int dims[4] = {1, 1, 1, 1};
      for (int i = 0; i < shape.size(); ++i)
        dims[i] = shape[i];
      ensure(CUDNN_STATUS_SUCCESS == cudnnCreateTensorDescriptor(&dataTensor));
      ensure(CUDNN_STATUS_SUCCESS == cudnnSetTensor4dDescriptor(dataTensor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        dims[0], dims[1], dims[2], dims[3]));
    }

    ~TensorHandler() {
      ensure(CUDNN_STATUS_SUCCESS == cudnnDestroyTensorDescriptor(dataTensor));
    }

    cudnnTensorDescriptor_t get() const {
      return dataTensor;
    }
  };


  static string stringify_shape(const vector<int> &shape, int offset = 0) {
    string ans = "(";
    if (offset == shape.size())
      return ans + ")";
    for (int i = offset; i < shape.size(); ++i)
      ans += to_string(shape[i]) + ((i + 1 < shape.size()) ? ", " : ")");
    return ans;
  }


  Tensor() {
    setup_tensor({0});
  }

  Tensor(const vector<int> &shape, bool random_fill = false) {
    size_t len = setup_tensor(shape);

    if (!random_fill)
      return;

    int fan_in, fan_out;
    if (shape.size() == 2)
      fan_in = shape[0], fan_out = shape[1];
    else {
      die_if(shape.size() != 4, "Not supporting random_fill for tensor of dimension = %zd.", shape.size());
      fan_in = shape[0] * shape[1] * shape[2];
      fan_out = shape[0] * shape[1] * shape[3];
    }

    float limit = sqrt(6.0f / (fan_in + fan_out));

    auto random_uniform = [&]() {
      // std::default_random_engine generator(time(0));
      // std::normal_distribution<float> normal(0.0f, 1.0f);
      // vector<float> r(len);
      // for (int i = 0; i < r.size(); ++i)
      //   r[i] = normal(generator);

      unsigned int seed = len;
      vector<float> r(len);
      for (int i = 0; i < r.size(); ++i)
        r[i] = (u_rand(&seed) / double(INT_MAX) - 0.5) * 2.0 * limit;
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

    ensure(sizeof(float) == sizeof(unsigned int));
    unsigned int ui = (unsigned int&)val;
    ensure(CUDA_SUCCESS == cuMemsetD32Async((CUdeviceptr)d_data->get(), ui, len, devices[currentDev].hStream));
  }


  size_t setup_tensor(const vector<int> &shape) {
    this->shape = shape;
    this->device = currentDev;

    size_t len = count();
    if (!len)
      return len;
    this->trainable = true;
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

  void set_data(const float *host, bool sync = true) const {
    size_t len = count();
    ensure(CUDA_SUCCESS == cuMemcpyHtoDAsync_v2((CUdeviceptr)d_data->get(), host, len * sizeof(float), devices[currentDev].hStream));
    if (sync)
      ensure(CUDA_SUCCESS == cuStreamSynchronize(devices[currentDev].hStream));
  }

  vector<float> get_data(bool sync = true) const {
    size_t len = count();
    vector<float> host(len);
    if (len > 0) {
      ensure(CUDA_SUCCESS == cuMemcpyDtoHAsync_v2(host.data(), (CUdeviceptr)d_data->get(), len * sizeof(float), devices[currentDev].hStream));
      if (sync)
        ensure(CUDA_SUCCESS == cuStreamSynchronize(devices[currentDev].hStream));
    }
    return move(host);
  }

  Tensor reshape(const vector<int> &shape, bool weak = false) const {
    Tensor mat = *this;
    mat.shape = shape;
    mat.dataTensor = make_shared<TensorHandler>(shape);
    if (!weak)
      ensure(mat.count() == count());
    return move(mat);
  }

  void copyTo(const Tensor &dst) const {
    die_if(dst.shape != this->shape, "Cannot copy tensor among two tensors with different shapes.");
    ensure(CUDA_SUCCESS == cuMemcpyDtoDAsync_v2((CUdeviceptr)dst.d_data->get(), (CUdeviceptr)this->d_data->get(), dst.count() * sizeof(float), devices[currentDev].hStream));
  }

  Tensor matmul(const Tensor &that, bool transposeThis = false, bool transposeThat = false) const {
    // ans = &that * this;
    const Tensor *A = &that, *B = this;
    bool transposeA = transposeThat, transposeB = transposeThis;

    ensure(A->shape.size() == 2 && B->shape.size() == 2);

    int ax = A->shape[1], ay = A->shape[0];
    if (transposeA)
      swap(ax, ay);
    int bx = B->shape[1], by = B->shape[0];
    if (transposeB)
      swap(bx, by);
    ensure(ay == bx);

    Tensor ans({by, ax});

    float alpha = 1.0f, beta = 0.0f;
    ensure(0 == cublasSgemm(cublasHandle,
                            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                            ax, by, ay, &alpha,
                            (float*)A->d_data->get(), A->shape[1],  // X
                            (float*)B->d_data->get(), B->shape[1],  // Y
                            &beta, (float*)ans.d_data->get(), ans.shape[1]));  // Z
    return ans;
  }

  float energy() const {
    double ans = 0.0;
    auto d = this->get_data();
    for (auto it: d)
      ans += it * it;
    return ans;
  }

  Tensor self_update(const Tensor &that, float alpha = 1.0f, float beta = 0.0f) const {
    if (fabs(alpha) < 1e-7f) {
      ensure(CUBLAS_STATUS_SUCCESS == cublasSscal(cublasHandle, count(), &beta, (float*)this->d_data->get(), 1));
      return *this;
    }
    ensure(this->shape == that.shape);
    ensure(CUDNN_STATUS_SUCCESS == cudnnTransformTensor(cudnnHandle,
        &alpha, that.dataTensor->get(), (float*)that.d_data->get(),
        &beta, this->dataTensor->get(), (float*)this->d_data->get()));
    return *this;
  }

  Tensor self_mul(float alpha) const {
    if (fabs(alpha - 1.0f) < 1e-7f)
      return *this;
    return self_update({}, 0.0f, alpha);
  }

  Tensor self_add(const Tensor &that, float ceof = 1.0f) const {
    return self_update(that, ceof, 1.0f);
  }

  Tensor add(const Tensor &that, float ceof = 1.0f) const {
    ensure(this->shape == that.shape);
    Tensor ans(this->shape, 0.0f);
    ensure(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &ceof, (float*)this->d_data->get(), 1, (float*)ans.d_data->get(), 1));
    // Tensor ans = this->copy();
    ensure(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, count(), &ceof, (float*)that.d_data->get(), 1, (float*)ans.d_data->get(), 1));
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
    ensure(CUDNN_STATUS_SUCCESS == cudnnOpTensor(cudnnHandle, op_desc,
      &alpha, this->dataTensor->get(), (float*)left.d_data->get(),
      &alpha, this->dataTensor->get(), (float*)this->d_data->get(),
      &beta, this->dataTensor->get(), (float*)interm.d_data->get()));

    cudnnSetOpTensorDescriptor(op_desc, CUDNN_OP_TENSOR_MIN, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
    ensure(CUDNN_STATUS_SUCCESS == cudnnOpTensor(cudnnHandle, op_desc,
      &alpha, this->dataTensor->get(), (float*)right.d_data->get(),
      &alpha, this->dataTensor->get(), (float*)interm.d_data->get(),
      &beta, this->dataTensor->get(), (float*)left.d_data->get()));
    cudnnDestroyOpTensorDescriptor(op_desc);
    return left;
  }

  pair<float, float> get_loss_and_accuracy_with(const Tensor &data_label) {
    const Tensor &data_pred = *this;
    ensure(data_pred.shape.size() == 2 && data_pred.shape == data_label.shape);

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

  shared_ptr<DeviceMemory> d_data;
  shared_ptr<TensorHandler> dataTensor;
  vector<int> shape;
  bool trainable;
  int device;
};

#endif
