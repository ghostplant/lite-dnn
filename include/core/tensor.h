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

#define rand_uniform_0_1(seed) ((u_rand(seed) + 1.0) / (RAND_MAX + 1.0))
#define rand_normal_0_1(seed)  sqrt(-2.0 * log(rand_uniform_0_1(seed))) * cos(2.0 * M_PI * rand_uniform_0_1(seed))

struct DeviceResources {
  CUstream hStream;
  CUcontext hContext;
  cudnnHandle_t hCudnn;
  cublasHandle_t hCublas;
};

static vector<DeviceResources> devices;

static vector<unordered_map<size_t, vector<void*>>> cached_mem;
static int currentDev = -1, globalStop;
static volatile int activeThread = 0;


static int mpi_size, mpi_rank, mpi_localrank;
static ncclComm_t comm;

class Tensor {

public:
  // Global Tensor Funtions

  static void init() {
    ensure(MPI_SUCCESS == MPI_Init(0, 0));
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    const char *localrank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    die_if(mpi_size > 1 && localrank == nullptr, "Only OpenMPI is supported for this application.");
    mpi_localrank = localrank ? atoi(localrank) : 0;

    int devCount = 0;
    ensure(CUDA_SUCCESS == cuInit(0));
    ensure(CUDA_SUCCESS == cuDeviceGetCount(&devCount));
    die_if(devCount <= 0 || mpi_localrank >= devCount, "No available GPUs detected for device rank = %d.", mpi_localrank);

    devCount = 1;
    devices.resize(devCount);

    for (int i = 0; i < devCount; ++i) {
      ensure(CUDA_SUCCESS == cuDevicePrimaryCtxRetain(&devices[i].hContext, mpi_localrank));
      ensure(CUDA_SUCCESS == cuCtxSetCurrent(devices[i].hContext));
      ensure(CUDA_SUCCESS == cuStreamCreate(&devices[i].hStream, CU_STREAM_NON_BLOCKING));
      ensure(CUBLAS_STATUS_SUCCESS == cublasCreate(&devices[i].hCublas));
      ensure(CUBLAS_STATUS_SUCCESS == cublasSetPointerMode_v2(devices[i].hCublas, CUBLAS_POINTER_MODE_HOST));
      ensure(CUDNN_STATUS_SUCCESS == cudnnCreate(&devices[i].hCudnn));
      ensure(CUBLAS_STATUS_SUCCESS == cublasSetStream_v2(devices[i].hCublas, devices[i].hStream));
      ensure(CUDNN_STATUS_SUCCESS == cudnnSetStream(devices[i].hCudnn, devices[i].hStream));
    }

    currentDev = 0;
    // devices[currentDev].hCublas = devices[currentDev].hCublas;
    // devices[currentDev].hCudnn = devices[currentDev].hCudnn;

    ncclUniqueId id;
    if (mpi_rank == 0)
      ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ensure(0 == ncclGroupStart());
    ensure(0 == ncclCommInitRank(&comm, mpi_size, id, mpi_rank));
    ensure(0 == ncclGroupEnd());
  }

  static void quit(int exitCode = 0) {
    globalStop = true;
    while (activeThread)
      usleep(50000);

    ensure(0 == ncclCommDestroy(comm));
    MPI_Finalize();
    exit(exitCode);
  }

  static void synchronizeCurrentDevice() {
    ensure(CUDA_SUCCESS == cuStreamSynchronize(devices[currentDev].hStream));
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
      synchronizeCurrentDevice();
  }

  vector<float> get_data(bool sync = true) const {
    size_t len = count();
    vector<float> host(len);
    if (len > 0) {
      ensure(CUDA_SUCCESS == cuMemcpyDtoHAsync_v2(host.data(), (CUdeviceptr)d_data->get(), len * sizeof(float), devices[currentDev].hStream));
      if (sync)
        synchronizeCurrentDevice();
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
    ensure(0 == cublasSgemm(devices[currentDev].hCublas,
                            transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
                            ax, by, ay, &alpha,
                            (float*)A->d_data->get(), A->shape[1],  // X
                            (float*)B->d_data->get(), B->shape[1],  // Y
                            &beta, (float*)ans.d_data->get(), ans.shape[1]));  // Z
    return ans;
  }

  void allreduce() const {
    ensure(0 == ncclAllReduce((const void*)this->d_data->get(), (void*)this->d_data->get(), this->count(), ncclFloat, ncclSum, comm, devices[currentDev].hStream));
  }

  double energy() const {
    double ans = 0.0;
    auto d = this->get_data();
    for (auto it: d)
      ans += it * it;
    return ans;
  }

  Tensor self_update(const Tensor &that, float alpha = 1.0f, float beta = 0.0f) const {
    if (fabs(alpha) < 1e-7f) {
      ensure(CUBLAS_STATUS_SUCCESS == cublasSscal(devices[currentDev].hCublas, count(), &beta, (float*)this->d_data->get(), 1));
      return *this;
    }
    ensure(this->shape == that.shape);
    ensure(CUDNN_STATUS_SUCCESS == cudnnTransformTensor(devices[currentDev].hCudnn,
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
    ensure(CUBLAS_STATUS_SUCCESS == cublasSaxpy(devices[currentDev].hCublas, count(), &ceof, (float*)this->d_data->get(), 1, (float*)ans.d_data->get(), 1));
    // Tensor ans = this->copy();
    ensure(CUBLAS_STATUS_SUCCESS == cublasSaxpy(devices[currentDev].hCublas, count(), &ceof, (float*)that.d_data->get(), 1, (float*)ans.d_data->get(), 1));
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
    ensure(CUDNN_STATUS_SUCCESS == cudnnOpTensor(devices[currentDev].hCudnn, op_desc,
      &alpha, this->dataTensor->get(), (float*)left.d_data->get(),
      &alpha, this->dataTensor->get(), (float*)this->d_data->get(),
      &beta, this->dataTensor->get(), (float*)interm.d_data->get()));

    cudnnSetOpTensorDescriptor(op_desc, CUDNN_OP_TENSOR_MIN, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN);
    ensure(CUDNN_STATUS_SUCCESS == cudnnOpTensor(devices[currentDev].hCudnn, op_desc,
      &alpha, this->dataTensor->get(), (float*)right.d_data->get(),
      &alpha, this->dataTensor->get(), (float*)interm.d_data->get(),
      &beta, this->dataTensor->get(), (float*)left.d_data->get()));
    cudnnDestroyOpTensorDescriptor(op_desc);
    return left;
  }

  unordered_map<string, float> compute_loss_and_accuracy(const Tensor &labels) {
    const Tensor &logits = *this;
    ensure(logits.shape.size() == 2 && logits.shape == labels.shape);

    vector<float> logit_data = logits.clip_by_value(1.0e-7f, 1.0f - 1.0e-7f).get_data();
    vector<float> label_data = labels.get_data();

    float loss = 0.0f;
    for (int i = 0; i < logit_data.size(); ++i) {
      loss -= label_data[i] * log(logit_data[i]) + (1.0f - label_data[i]) * log(1.0f - logit_data[i]);
    }
    loss /= logits.shape[0];

    int tot = 0, acc1 = 0, acc5 = 0;
    for (int i = 0; i < logits.shape[0]; ++i) {
      int it = 0, jt = 0;
      for (int j = 1; j < logits.shape[1]; ++j) {
        if (label_data[i * logits.shape[1] + jt] < label_data[i * logits.shape[1] + j])
          jt = j;
      }
      for (int j = 0; j < logits.shape[1]; ++j) {
        if (logit_data[i * logits.shape[1] + jt] <= logit_data[i * logits.shape[1] + j])
          ++it;
      }
      ++tot;
      if (it <= 1)
        ++acc1;
      if (it <= 5)
        ++acc5;
    }
    return {
      {"loss", loss},
      {"top_1_acc", acc1 * 100.0f / tot},
      {"top_5_acc", acc5 * 100.0f / tot},
    };
  }

  shared_ptr<DeviceMemory> d_data;
  shared_ptr<TensorHandler> dataTensor;
  vector<int> shape;
  bool trainable;
  int device;
};

#endif
