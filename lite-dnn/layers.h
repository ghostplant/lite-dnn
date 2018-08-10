#ifndef __LITEDNN_LAYERS__
#define __LITEDNN_LAYERS__


class Layer {

public:
  virtual string to_string() const = 0;

  virtual Tensor forward(const Tensor &x) = 0;

  virtual Tensor backward(const Tensor &dy) = 0;


  virtual vector<Tensor> get_weights() const {
    return {};
  }

  virtual vector<Tensor> get_gradients(const Tensor &dy) const {
    return {};
  }

  virtual vector<int> configure(const vector<int> &input_shape) {
    return input_shape;
  }

  /*
  vector<shared_ptr<Layer>> parents;
  template <typename... Arguments> void from(Arguments... args) {
    this->parents = {std::forward<Arguments>(args)...};
  }
  */

protected:
  vector<Tensor> cacheTensors;
};


class InputLayer: public Layer {

public:
  vector<int> input_shape;

  InputLayer(int channel, int height = -1, int width = -1): input_shape({-1, channel}) {
    if (height > 0 && width > 0)
      input_shape.push_back(height), input_shape.push_back(width);
  }

  ~InputLayer() {
  }

  vector<int> configure(const vector<int> &input_shape = {}) {
    if (input_shape.size() >= 1)
      this->input_shape[0] = input_shape[0];
    return this->input_shape;
  }

  string to_string() const {
    return "InputLayer";
  }

  Tensor forward(const Tensor &x) {
    int batch = input_shape[0];
    input_shape[0] = x.shape[0];
    die_if(input_shape != x.shape, "The input image shape %s doesn't match the shape of input layer.", Tensor::stringify_shape(x.shape, 1).c_str());
    input_shape[0] = batch;
    return x;
  }

  Tensor backward(const Tensor &dy) {
    return {};
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

    Tensor y(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    cacheTensors = {y};
    return y;
  }

  Tensor backward(const Tensor &dy) {
    const Tensor &y = cacheTensors[0];
    assert(dy.shape == y.shape);

    float posi = 1.0f / dy.shape[0], nega = -1.0f / dy.shape[0];
    size_t len = dy.count();

    Tensor dx(dy.shape, 0.0f);
    cublasSaxpy(cublasHandle, len, &posi, (float*)y.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    cublasSaxpy(cublasHandle, len, &nega, (float*)dy.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    return move(dx);
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
    Tensor y(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationForward(cudnnHandle, activationDesc, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    cacheTensors = {x, y};
    return move(y);
  }

  Tensor backward(const Tensor &dy) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx = dy;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationBackward(cudnnHandle, activationDesc, &alpha, y.dataTensor->get(), (float*)y.d_data->get(),
        dy.dataTensor->get(), (float*)dy.d_data->get(), x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
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

    cacheTensors = {x, y};
    return move(y);
  }

  Tensor backward(const Tensor &dy) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelBackward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
        x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
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

    cacheTensors = {x, y};
    return move(y);
  }

  Tensor backward(const Tensor &dy) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingBackward(cudnnHandle, poolDesc, &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
                         x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
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

    Tensor y(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutForward(cudnnHandle, dropDesc, x.dataTensor->get(), (float*)x.d_data->get(),
      y.dataTensor->get(), (float*)y.d_data->get(), reversed->get(), reversed_size));
    cacheTensors = {x, y};
    return move(y);
  }

  Tensor backward(const Tensor &dy) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];
    assert(CUDNN_STATUS_SUCCESS == cudnnRestoreDropoutDescriptor(dropDesc, cudnnHandle, drop_prob, states->get(), states_size, seed));

    size_t _reversed_size;
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(y.dataTensor->get(), &_reversed_size));
    assert(_reversed_size <= reversed_size);

    Tensor dx(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutBackward(cudnnHandle, dropDesc, dy.dataTensor->get(), (float*)dy.d_data->get(),
      dx.dataTensor->get(), (float*)dx.d_data->get(), reversed->get(), reversed_size));
    return move(dx);
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
    cacheTensors = {x};
    return x.reshape(this->configure(x.shape));
  }

  Tensor backward(const Tensor &dy) {
    const Tensor &x = cacheTensors[0];
    return dy.reshape(x.shape);
  }
};


class Dense: public Layer {

public:
  Tensor w, bias, ones;
  int channels;

  Dense(int channels, int max_batch = 1024): channels(channels), ones({max_batch, 1}, 1.0f), w(), bias() {
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

    auto y = x.matmul(w, false, false);
    // y = x * w + ones * bias';
    assert(y.shape.size() == 2 && bias.shape.size() == 2 && y.shape[1] == bias.shape[1] && y.shape[0] <= ones.shape[0]);
    // auto wx_b = y.add(ones.reshape({x.shape[0], 1}, true).matmul(bias, false, false));
    // return move(wx_b);

    float alpha = 1.0f;
    cublasSgemm(cublasHandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                bias.shape[1], y.shape[0], 1,
                &alpha,
                (float*)bias.d_data->get(), bias.shape[1],  // B
                (float*)ones.d_data->get(), 1,  // 1
                &alpha,
                (float*)y.d_data->get(), y.shape[1]);  // self
    cacheTensors = {x, y};
    return move(y);
  }

  Tensor backward(const Tensor &dy) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];
    assert(dy.shape == y.shape);
    // dx = dy * w'
    return dy.matmul(w, false, true);
  }

  vector<Tensor> get_gradients(const Tensor &dy) const {
    const Tensor &x = cacheTensors[0];
    return {
      x.matmul(dy, true, false),  // dw = x' * dy
      ones.reshape({x.shape[0], 1}, true).matmul(dy, true, false)  // db = sum(dy, axis=0)
    };
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

  Convolution(int filters, int kernel_size, int stride = 1, int padding = 0, bool use_bias = false):
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
      if (use_bias) {
        w_bias = Tensor({1, filters, 1, 1}, 0.0f);
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

    Tensor y(output_shape);

    cudnnConvolutionFwdAlgo_t convalgo;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardAlgorithm(cudnnHandle, x.dataTensor->get(), filterDesc, convDesc,
        y.dataTensor->get(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convalgo));

    size_t sizeInBytes;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, x.dataTensor->get(), filterDesc, convDesc,
        y.dataTensor->get(), convalgo, &sizeInBytes));

    DeviceMemory workspace(sizeInBytes);

    assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionForward(cudnnHandle, &alpha, x.dataTensor->get(), (float*)x.d_data->get(), filterDesc,
        (float*)w_krnl.d_data->get(), convDesc, convalgo, workspace.get(), sizeInBytes, &beta, y.dataTensor->get(), (float*)y.d_data->get()));
    if (use_bias)
      assert(CUDNN_STATUS_SUCCESS == cudnnAddTensor(cudnnHandle,
        &alpha, w_bias.dataTensor->get(), (float*)w_bias.d_data->get(), &alpha, y.dataTensor->get(), (float*)y.d_data->get()));
    cacheTensors = {x, y};
    return move(y);
  }

  Tensor backward(const Tensor &dy) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];
    float alpha = 1.0f, beta = 0.0f;
    assert(y.shape[1] == filters);
    int n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];

    cudnnConvolutionBwdDataAlgo_t dalgo;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardDataAlgorithm(
                cudnnHandle, filterDesc, y.dataTensor->get(), convDesc, x.dataTensor->get(),
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &dalgo));

    size_t sizeInBytes;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardDataWorkspaceSize(
                cudnnHandle, filterDesc, y.dataTensor->get(), convDesc, x.dataTensor->get(), 
                dalgo, &sizeInBytes));

    DeviceMemory workspace(sizeInBytes);

    Tensor dx({n, c, h, w});
    assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardData(cudnnHandle, &alpha,
                filterDesc, (float*)w_krnl.d_data->get(),
                dy.dataTensor->get(), (float*)dy.d_data->get(),
                convDesc, dalgo, workspace.get(), sizeInBytes, &beta,
                dx.dataTensor->get(), (float*)dx.d_data->get()));
    return move(dx);
  }

  vector<Tensor> get_gradients(const Tensor &dy) const {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    vector<Tensor> grads = { Tensor(w_krnl.shape) };

    cudnnConvolutionBwdFilterAlgo_t falgo;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnnHandle, x.dataTensor->get(), y.dataTensor->get(), convDesc, filterDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &falgo));

    size_t sizeInBytes;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardFilterWorkspaceSize(
                cudnnHandle, x.dataTensor->get(), y.dataTensor->get(), convDesc, filterDesc, 
                falgo, &sizeInBytes));

    DeviceMemory workspace(sizeInBytes);

    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardFilter(cudnnHandle, &alpha,
                x.dataTensor->get(), (float*)x.d_data->get(),
                dy.dataTensor->get(), (float*)dy.d_data->get(),
                convDesc, falgo, workspace.get(), sizeInBytes, &beta,
                filterDesc, (float*)grads[0].d_data->get()));


    if (use_bias) {
      grads.push_back(Tensor(w_bias.shape));

      assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardBias(cudnnHandle, &alpha,
             dy.dataTensor->get(), (float*)dy.d_data->get(), &beta,
             grads[1].dataTensor->get(), (float*)grads[1].d_data->get()));
    }
    return move(grads);
  }

  vector<Tensor> get_weights() const {
    vector<Tensor> weights = {w_krnl};
    if (use_bias)
      weights.push_back(w_bias);
    return move(weights);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void model_configure_shape(auto &model) {
  puts("");
  vector<int> val_shape;
  for (int i = 0; i < model.size(); ++i) {
    val_shape = model[i]->configure(val_shape);
    printf("layer: %20s, output_shape = %s\n", model[i]->to_string().c_str(), Tensor::stringify_shape(val_shape, 1).c_str());
  }
  puts("");
}

bool model_load_weights(auto &model, const char *weight_path) {
  FILE *fp = fopen(weight_path, "rb");
  if (fp == nullptr)
    return false;

  puts("  [@] Loading saved weights ..");
  ssize_t w_count = 0;
  for (auto &layer: model) {
    auto sym_weights = layer->get_weights();
    for (auto &weight: sym_weights)
      w_count += weight.count();
  }
  fseek(fp, 0, SEEK_END);
  ssize_t f_bytes = ftell(fp);

  die_if(f_bytes != w_count * sizeof(float), "The weight file `%s` doesn't match the current model.", weight_path);
  fseek(fp, 0, SEEK_SET);

  for (auto &layer: model) {
    auto sym_weights = layer->get_weights();
    for (auto &weight: sym_weights) {
      vector<float> host(weight.count());
      assert(host.size() == fread(host.data(), sizeof(float), host.size(), fp));
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
