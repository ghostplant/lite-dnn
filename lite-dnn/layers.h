#ifndef __LITEDNN_LAYERS__
#define __LITEDNN_LAYERS__


class Layer {

public:
  virtual string to_string() const = 0;

  virtual Tensor forward(const Tensor &x) = 0;

  virtual Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) = 0;

  virtual void learn(float lr) const = 0;

  virtual vector<Tensor> get_weights() const = 0;

  virtual vector<int> configure(const vector<int> &input_shape) {
    return input_shape;
  }
};


class SoftmaxCrossEntropy: public Layer {

public:
  SoftmaxCrossEntropy() {
  }

  ~SoftmaxCrossEntropy() {
  }

  string to_string() const {
    return "SoftmaxCrossEntropy";
  }

  Tensor forward(const Tensor &x) {
    float alpha = 1.0f, beta = 0.0f;

    Tensor ans(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, ans.dataTensor->get(), (float*)ans.d_data->get()));
    return ans;
  }

  Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) {
    assert(x.shape == y.shape);

    float posi = 1.0f / x.shape[0], nega = -1.0f / x.shape[0], batch = 1.0f / x.shape[0];
    size_t len = x.count();

    // Tensor dx = y;
    // assert(CUDNN_STATUS_SUCCESS == cudnnAddTensor(cudnnHandle,
    //   &posi, this->dataTensor->get(), (float*)this->d_data->get(), &nega, dx.dataTensor->get(), (float*)dx.d_data->get()));

    Tensor dx(x.shape, 0.0f);
    cublasSaxpy(cublasHandle, len, &posi, (float*)x.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    cublasSaxpy(cublasHandle, len, &nega, (float*)y.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    return move(dx);

    /*if (lastLayer)
      return dy;

    Tensor dx(x.shape, 0.0f);
    // float alpha = -float(count() / this->shape[0]) / this->shape[0], beta = 0.0f;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxBackward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
      &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
      &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return dx;*/
  }

  void learn(float lr) const {
  }

  vector<Tensor> get_weights() const {
    return {};
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

  string to_string() const {
    return "Activation";
  }

  Tensor forward(const Tensor &x) {
    Tensor ans(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationForward(cudnnHandle, activationDesc, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, ans.dataTensor->get(), (float*)ans.d_data->get()));
    return move(ans);
  }

  Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) {
    if (lastLayer)
      return dy;
    Tensor dx = dy;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationBackward(cudnnHandle, activationDesc, &alpha, y.dataTensor->get(), (float*)y.d_data->get(),
        dy.dataTensor->get(), (float*)dy.d_data->get(), x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }

  void learn(float lr) const {
  }

  vector<Tensor> get_weights() const {
    return {};
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

  string to_string() const {
    return "LRN";
  }

  Tensor forward(const Tensor &x) {
    assert(x.shape.size() == 4);

    Tensor y(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelForward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));
    return move(y);
  }

  Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) {
    Tensor dx(x.shape);
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

  vector<Tensor> get_weights() const {
    return {};
  }
};


class Pooling: public Layer {
  int size, stride;
  cudnnPoolingDescriptor_t poolDesc;

public:
  Pooling(int size, int stride, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX): size(size), stride(stride) {
    assert(stride > 0);
    assert(CUDNN_STATUS_SUCCESS == cudnnCreatePoolingDescriptor(&poolDesc));
    assert(CUDNN_STATUS_SUCCESS == cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN, size, size, 0, 0, stride, stride));
  }

  ~Pooling() {
    assert(CUDNN_STATUS_SUCCESS == cudnnDestroyPoolingDescriptor(poolDesc));
  }

  string to_string() const {
    return "Pooling";
  }

  vector<int> configure(const vector<int> &input_shape) {
    die_if(input_shape.size() != 4, "Currently Pooling Layer only suport 4D tensor.");
    return {input_shape[0], input_shape[1], (input_shape[2] - (size - stride)) / stride, (input_shape[3] - (size - stride)) / stride};
  }

  Tensor forward(const Tensor &x) {
    assert(x.shape.size() == 4);
    // assert((x.shape[2] - (size - stride)) % stride == 0 && (x.shape[3] - (size - stride)) % stride == 0);

    Tensor y(this->configure(x.shape));
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingForward(cudnnHandle, poolDesc, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));
    return move(y);
  }

  Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) {
    if (lastLayer)
      return dy;
    Tensor dx(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
                         x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }

  void learn(float lr) const {
  }

  vector<Tensor> get_weights() const {
    return {};
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

  string to_string() const {
    return "Dropout";
  }

  Tensor forward(const Tensor &x) {
    size_t _reversed_size;
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(x.dataTensor->get(), &_reversed_size));

    if (reversed_size == ~0LU || _reversed_size > reversed_size) {
      reversed_size = _reversed_size;
      reversed = make_shared<DeviceMemory>(reversed_size);
    }

    Tensor ans(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutForward(cudnnHandle, dropDesc, x.dataTensor->get(), (float*)x.d_data->get(),
      ans.dataTensor->get(), (float*)ans.d_data->get(), reversed->get(), reversed_size));
    return move(ans);
  }

  Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) {
    assert(CUDNN_STATUS_SUCCESS == cudnnRestoreDropoutDescriptor(dropDesc, cudnnHandle, drop_prob, states->get(), states_size, seed));

    size_t _reversed_size;
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(y.dataTensor->get(), &_reversed_size));
    assert(_reversed_size <= reversed_size);

    if (lastLayer)
      return dy;
    Tensor dx(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutBackward(cudnnHandle, dropDesc, dy.dataTensor->get(), (float*)dy.d_data->get(),
      dx.dataTensor->get(), (float*)dx.d_data->get(), reversed->get(), reversed_size));
    return move(dx);
  }

  void learn(float lr) const {
  }

  vector<Tensor> get_weights() const {
    return {};
  }
};


class Flatten: public Layer {

public:
  Flatten() {
  }

  vector<int> configure(const vector<int> &input_shape) {
    int count = 1;
    for (int i = 1; i < input_shape.size(); ++i)
      count *= input_shape[i];
    return {input_shape[0], count};
  }

  string to_string() const {
    return "Flatten";
  }

  Tensor forward(const Tensor &x) {
    return x.reshape(this->configure(x.shape));
  }

  Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) {
    return dy.reshape(x.shape);
  }

  void learn(float lr) const {
  }

  vector<Tensor> get_weights() const {
    return {};
  }
};


class Dense: public Layer {

public:
  Tensor w, bias, ones, g_bias, g_w;
  int channels;

  Dense(int channels, int max_batch = 1024): channels(channels), ones({max_batch, 1}, 1.0f), g_bias(), g_w(), w(), bias() {
  }

  vector<int> configure(const vector<int> &input_shape) {
    if (w.count() < 1) {
      assert(input_shape.size() == 2);
      w = Tensor({input_shape[1], channels}, true);
      bias = Tensor({1, channels}, 0.0f);
    }
    return {input_shape[0], channels};
  }

  string to_string() const {
    return "Dense";
  }

  Tensor forward(const Tensor &x) {
    assert(x.shape.size() == 2);

    auto out = x.matmul(w, false, false);
    // y = x * w + ones * bias';
    assert(out.shape.size() == 2 && bias.shape.size() == 2 && out.shape[1] == bias.shape[1] && out.shape[0] <= ones.shape[0]);
    // auto wx_b = out.add(ones.reshape({x.shape[0], 1}, true).matmul(bias, false, false));
    // return move(wx_b);

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

  Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) {
    assert(dy.shape == y.shape);
    // dw = x' * dy
    g_w = x.matmul(dy, true, false);
    g_bias = ones.reshape({x.shape[0], 1}, true).matmul(dy, true, false);
    if (lastLayer)
      return dy;
    // dx = dy * w'
    return dy.matmul(w, false, true);
  }

  void learn(float lr) const {
    assert(w.shape == g_w.shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, w.count(), &lr, (float*)g_w.d_data->get(), 1, (float*)w.d_data->get(), 1));

    assert(bias.shape == g_bias.shape);
    assert(CUBLAS_STATUS_SUCCESS == cublasSaxpy(cublasHandle, bias.count(), &lr, (float*)g_bias.d_data->get(), 1, (float*)bias.d_data->get(), 1));
  }

  vector<Tensor> get_weights() const {
    return {w, bias};
  }
};


class Convolution: public Layer {

public:
  int filters, kernel_size, stride, padding;
  Tensor w_krnl, w_bias, g_krnl, g_bias;
  bool use_bias;

  cudnnConvolutionDescriptor_t convDesc;
  cudnnFilterDescriptor_t filterDesc;

  Convolution(int filters, int kernel_size, int stride = 1, int padding = 0, bool use_bias = false): w_krnl(), w_bias(), g_krnl(), g_bias(),
      filters(filters), kernel_size(kernel_size), stride(stride), padding(padding), convDesc(NULL), filterDesc(NULL), use_bias(use_bias) {
  }

  ~Convolution() {
    if (filterDesc)
      assert(CUDNN_STATUS_SUCCESS == cudnnDestroyFilterDescriptor(filterDesc));
    if (convDesc)
      assert(CUDNN_STATUS_SUCCESS == cudnnDestroyConvolutionDescriptor(convDesc));
  }

  vector<int> configure(const vector<int> &input_shape) {
    if (w_krnl.count() < 1) {
      w_krnl = Tensor({kernel_size, kernel_size, input_shape[1], filters}, true);
      g_krnl = Tensor(w_krnl.shape);
      if (use_bias) {
        w_bias = Tensor({1, filters, 1, 1}, 0.0f);
        g_bias = Tensor(w_bias.shape);
      }
      cudnnCreateConvolutionDescriptor(&convDesc);
      cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
      // assert(CUDNN_STATUS_SUCCESS == cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

      cudnnCreateFilterDescriptor(&filterDesc);
      cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          filters, input_shape[1], kernel_size, kernel_size);
    }
    int nn = input_shape[0], cc = filters, hh = (input_shape[2] + padding + padding - max(0, kernel_size - stride)) / stride, ww = (input_shape[3] + padding + padding - max(0, kernel_size - stride)) / stride;
    return {nn, cc, hh, ww};
  }

  string to_string() const {
    return "Convolution";
  }

  Tensor forward(const Tensor &x) {
    assert(x.shape.size() == 4);

    float alpha = 1.0f, beta = 0.0f;

    vector<int> output_shape = configure(x.shape), cu_shape(4);
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolution2dForwardOutputDim(convDesc, x.dataTensor->get(), filterDesc, \
        &cu_shape[0], &cu_shape[1], &cu_shape[2], &cu_shape[3]));
    assert(output_shape == cu_shape);

    Tensor ans(output_shape);

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

  Tensor backward(const Tensor &dy, const Tensor &y, const Tensor &x, bool lastLayer = false) {
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

    Tensor dx({n, c, h, w});
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

  vector<Tensor> get_weights() const {
    if (use_bias)
      return {w_krnl, w_bias};
    return {w_krnl};
  }
};


void model_configure_shape(auto &model, vector<int> input_shape) {
  puts("");
  for (int i = 0; i < model.size(); ++i) {
    input_shape = model[i]->configure(input_shape);
    printf("layer: %20s, output_shape = (", model[i]->to_string().c_str());
    for (int j = 1; j < input_shape.size(); ++j)
      printf("%d%s", input_shape[j], j + 1 < input_shape.size() ? ", ": ")\n");
  }
  puts("");
}

bool model_load_weights(auto &model, const char *weight_path) {
  FILE *fp = fopen(weight_path, "rb");
  if (fp == nullptr)
    return false;

  puts("  [@] Loading saved weights ..");
  for (auto &layer: model) {
    auto sym_weights = layer->get_weights();
    for (auto &weight: sym_weights) {
      vector<float> host(weight.count());
      die_if(host.size() != fread(host.data(), sizeof(float), host.size(), fp), "The file `weights.lw` doesn't match current model.");
      weight.set_data(host.data());
    }
  }
  fclose(fp);
  return true;
}

bool model_save_weights(auto &model, const char *weight_path) {
  FILE *fp = fopen("weights.lw", "wb");
  if (fp == nullptr)
    return false;

  puts("  [@] Saving saved weights ..");
  for (auto &layer: model) {
    auto sym_weights = layer->get_weights();
    for (auto &weight: sym_weights) {
      auto host = weight.get_data();
      assert(host.size() == fwrite(host.data(), sizeof(float), host.size(), fp));
    }
  }
  fclose(fp);
  return true;
}

#endif
