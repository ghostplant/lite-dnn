#ifndef __LITEDNN_LAYERS__
#define __LITEDNN_LAYERS__

class Model;


class Layer: public std::enable_shared_from_this<Layer> {

public:
  virtual string to_string() const = 0;

  virtual Tensor forward(const vector<Tensor> &x, const unordered_map<string, Tensor> &feed_dict) = 0;

  virtual vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) = 0;


  virtual vector<Tensor> get_weights() const {
    return {};
  }

  virtual vector<Tensor> get_gradients(const Tensor &dy) const {
    return {};
  }

  virtual vector<int> get_output_shape() {
    return this->input_shape;
  }

  /////////////////////////////////////////////////////

  shared_ptr<Layer> then(const shared_ptr<Layer> &that) {
    ensure(this->input_shape.size() > 0);
    that->parents = { shared_from_this() };
    that->input_shape = this->get_output_shape();
    that->depth = this->depth + 1;
    return that;
  }

  shared_ptr<Model> compile() {
    return make_shared<Model>(shared_from_this());
  }

  vector<shared_ptr<Layer>> parents;
  vector<Tensor> cacheTensors;
  vector<int> input_shape;
  int depth;
};


class InputLayer: public Layer {

public:
  string place_holder;

  InputLayer(const string &place_holder, int channel, int height = -1, int width = -1): place_holder(place_holder) {
    if (height > 0 && width > 0)
      input_shape = {-1, channel, height, width};
    else
      input_shape = {-1, channel};
    this->depth = 1;
  }

  ~InputLayer() {
  }

  string to_string() const {
    return "InputLayer";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    auto it = feed_dict.find(place_holder);
    die_if(it == feed_dict.end(), "Cannot find item `%s` in feed_dict.", place_holder.c_str());

    auto x_shape = it->second.shape;
    x_shape[0] = -1;
    die_if(input_shape != x_shape, "The shape of image fed doesn't match the expected shape of input layer: %s.",
        Tensor::stringify_shape(x_shape, 1).c_str());
    return it->second;
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    return {};
  }
};


class Concat: public Layer {

public:
  Concat(const auto &...a) {
    this->parents = {a...};
    die_if(this->parents.size() == 0, "Not allowed to have zero parent layer to add.");

    this->input_shape = this->parents[0]->get_output_shape();
    this->depth = this->parents[0]->depth;

    for (int i = 1; i < this->parents.size(); ++i) {
      die_if(this->input_shape != this->parents[i]->get_output_shape(), "Output shape for each parent layer doesn't match with each other.");
      this->depth = max(this->depth, this->parents[i]->depth);
    }

    ++this->depth;
  }

  ~Concat() {
  }

  string to_string() const {
    return "Concat";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    Tensor y = xs[0];
    for (int i = 1; i < xs.size(); ++i)
      y = y.add(xs[i]);
    return y;
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    return vector<Tensor>(parents.size(), dy);
  }
};


class SoftmaxCrossEntropy: public Layer {

public:
  string place_holder;

  SoftmaxCrossEntropy(const string &place_holder): place_holder(place_holder) {
  }

  ~SoftmaxCrossEntropy() {
  }

  string to_string() const {
    return "SoftmaxCrossEntropy";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    float alpha = 1.0f, beta = 0.0f;

    Tensor y(xs[0].shape);
    ensure(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(devices[currentDev].hCudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));
    cacheTensors = {y};
    return y;
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &y = cacheTensors[0];
    auto it = feed_dict.find(place_holder);
    die_if(it == feed_dict.end(), "Cannot find item `%s` in feed_dict.", place_holder.c_str());
    const Tensor &_dy = it->second;

    ensure(_dy.shape == y.shape);

    float posi = 1.0f / _dy.shape[0], nega = -1.0f / _dy.shape[0];
    size_t len = _dy.count();

    Tensor dx(_dy.shape, 0.0f);
    cublasSaxpy(devices[currentDev].hCublas, len, &posi, (float*)y.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    cublasSaxpy(devices[currentDev].hCublas, len, &nega, (float*)_dy.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    return { move(dx) };
  }
};


class Activation: public Layer {
  cudnnActivationDescriptor_t activationDesc;

public:
  Activation(cudnnActivationMode_t mode) {
    ensure(CUDNN_STATUS_SUCCESS == cudnnCreateActivationDescriptor(&activationDesc));
    ensure(CUDNN_STATUS_SUCCESS == cudnnSetActivationDescriptor(activationDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));
  }

  ~Activation() {
    ensure(CUDNN_STATUS_SUCCESS == cudnnDestroyActivationDescriptor(activationDesc));
  }

  string to_string() const {
    return "Activation";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    Tensor y(xs[0].shape);
    float alpha = 1.0f, beta = 0.0f;
    ensure(CUDNN_STATUS_SUCCESS == cudnnActivationForward(devices[currentDev].hCudnn, activationDesc, \
        &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    cacheTensors = {y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &y = cacheTensors[0];

    // ignore x value, reuse y value as dx
    Tensor dx = y; // Tensor dx(y.shape);
    float alpha = 1.0f, beta = 0.0f;
    ensure(CUDNN_STATUS_SUCCESS == cudnnActivationBackward(devices[currentDev].hCudnn, activationDesc,
        &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
        y.dataTensor->get(), (float*)y.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return { move(dx) };
  }
};


class LRN: public Layer {
  unsigned lrnN;
  float bias, lrnAlpha, lrnBeta;
  cudnnLRNDescriptor_t lrnDesc;

public:
  LRN(unsigned depthRadius, float bias, float lrnAlpha, float lrnBeta):
      lrnN(2 * depthRadius + 1), bias(bias), lrnAlpha(lrnAlpha), lrnBeta(lrnBeta) {
    ensure(bias >= CUDNN_LRN_MIN_K && lrnBeta >= CUDNN_LRN_MIN_BETA && lrnN >= CUDNN_LRN_MIN_N && lrnN <= CUDNN_LRN_MAX_N);
    ensure(CUDNN_STATUS_SUCCESS == cudnnCreateLRNDescriptor(&lrnDesc));
    ensure(CUDNN_STATUS_SUCCESS == cudnnSetLRNDescriptor(lrnDesc, lrnN, lrnAlpha * lrnN, lrnBeta, bias));
  }

  ~LRN() {
    ensure(CUDNN_STATUS_SUCCESS == cudnnDestroyLRNDescriptor(lrnDesc));
  }

  string to_string() const {
    return "LRN";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    ensure(xs[0].shape.size() == 4);

    Tensor y(xs[0].shape);
    float alpha = 1.0f, beta = 0.0f;
    ensure(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelForward(devices[currentDev].hCudnn, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    ensure(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelBackward(devices[currentDev].hCudnn, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
        x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return { move(dx) };
  }
};


class Pooling: public Layer {
  int size, stride;
  cudnnPoolingDescriptor_t poolDesc;

public:
  Pooling(int size, int stride, cudnnPoolingMode_t mode = CUDNN_POOLING_MAX): size(size), stride(stride) {
    ensure(stride > 0);
    ensure(CUDNN_STATUS_SUCCESS == cudnnCreatePoolingDescriptor(&poolDesc));
    ensure(CUDNN_STATUS_SUCCESS == cudnnSetPooling2dDescriptor(poolDesc, mode, CUDNN_PROPAGATE_NAN, size, size, 0, 0, stride, stride));
  }

  ~Pooling() {
    ensure(CUDNN_STATUS_SUCCESS == cudnnDestroyPoolingDescriptor(poolDesc));
  }

  string to_string() const {
    return "Pooling";
  }

  vector<int> get_output_shape() {
    die_if(input_shape.size() != 4, "Currently Pooling Layer only suport 4D tensor (NCHW).");
    return {input_shape[0], input_shape[1], (input_shape[2] - (size - stride)) / stride, (input_shape[3] - (size - stride)) / stride};
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    auto shape = this->get_output_shape();
    shape[0] = xs[0].shape[0];
    Tensor y(shape);
    float alpha = 1.0f, beta = 0.0f;
    ensure(CUDNN_STATUS_SUCCESS == cudnnPoolingForward(devices[currentDev].hCudnn, poolDesc,
        &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    ensure(CUDNN_STATUS_SUCCESS == cudnnPoolingBackward(devices[currentDev].hCudnn, poolDesc,
        &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
        x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return { move(dx) };
  }
};


class Dropout: public Layer {
  cudnnDropoutDescriptor_t dropDesc;

  shared_ptr<Tensor::DeviceMemory> states, reversed;
  size_t states_size, reversed_size;
  uint64_t seed; float drop_prob;

public:
  Dropout(float drop_prob = 0.1f, uint64_t seed = 10): reversed_size(~0LU), seed(seed), drop_prob(drop_prob) {
    ensure(CUDNN_STATUS_SUCCESS == cudnnCreateDropoutDescriptor(&dropDesc));
    ensure(CUDNN_STATUS_SUCCESS == cudnnDropoutGetStatesSize(devices[currentDev].hCudnn, &states_size));

    states = make_shared<Tensor::DeviceMemory>(states_size);
    ensure(CUDNN_STATUS_SUCCESS == cudnnSetDropoutDescriptor(dropDesc, devices[currentDev].hCudnn, drop_prob, states->get(), states_size, seed));
  }

  ~Dropout() {
    ensure(CUDNN_STATUS_SUCCESS == cudnnDestroyDropoutDescriptor(dropDesc));
  }

  string to_string() const {
    return "Dropout";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    size_t _reversed_size;
    ensure(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(xs[0].dataTensor->get(), &_reversed_size));

    if (reversed_size == ~0LU || _reversed_size > reversed_size) {
      reversed_size = _reversed_size;
      reversed = make_shared<Tensor::DeviceMemory>(reversed_size);
    }

    Tensor y(xs[0].shape);
    ensure(CUDNN_STATUS_SUCCESS == cudnnDropoutForward(devices[currentDev].hCudnn, dropDesc, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(),
      y.dataTensor->get(), (float*)y.d_data->get(), reversed->get(), reversed_size));
    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];
    ensure(CUDNN_STATUS_SUCCESS == cudnnRestoreDropoutDescriptor(dropDesc, devices[currentDev].hCudnn, drop_prob, states->get(), states_size, seed));

    size_t _reversed_size;
    ensure(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(y.dataTensor->get(), &_reversed_size));
    ensure(_reversed_size <= reversed_size);

    Tensor dx(x.shape);
    ensure(CUDNN_STATUS_SUCCESS == cudnnDropoutBackward(devices[currentDev].hCudnn, dropDesc, dy.dataTensor->get(), (float*)dy.d_data->get(),
      dx.dataTensor->get(), (float*)dx.d_data->get(), reversed->get(), reversed_size));
    return { move(dx) };
  }
};


class Flatten: public Layer {

public:
  Flatten() {
  }

  vector<int> get_output_shape() {
    int count = 1;
    for (int i = 1; i < input_shape.size(); ++i)
      count *= input_shape[i];
    return {input_shape[0], count};
  }

  string to_string() const {
    return "Flatten";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    cacheTensors = {xs[0]};
    auto shape = this->get_output_shape();
    shape[0] = xs[0].shape[0];
    return xs[0].reshape(shape);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0];
    return { dy.reshape(x.shape) };
  }
};


class Dense: public Layer {

public:
  Tensor w, bias, ones;
  int channels;

  Dense(int channels, int max_batch = 1024): channels(channels), ones({max_batch, 1}, 1.0f), w(), bias() {
  }

  vector<int> get_output_shape() {
    if (w.count() < 1) {
      ensure(input_shape.size() == 2);
      w = Tensor({input_shape[1], channels}, true);
      bias = Tensor({1, channels}, 0.0f);
    }
    return {input_shape[0], channels};
  }

  string to_string() const {
    return "Dense";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    ensure(xs[0].shape.size() == 2);

    auto y = xs[0].matmul(w, false, false);
    // y = xs[0] * w + ones * bias';
    ensure(y.shape.size() == 2 && bias.shape.size() == 2 && y.shape[1] == bias.shape[1] && y.shape[0] <= ones.shape[0]);
    // auto wx_b = y.add(ones.reshape({xs[0].shape[0], 1}, true).matmul(bias, false, false));
    // return move(wx_b);

    float alpha = 1.0f;
    cublasSgemm(devices[currentDev].hCublas,
                CUBLAS_OP_N, CUBLAS_OP_N,
                bias.shape[1], y.shape[0], 1,
                &alpha,
                (float*)bias.d_data->get(), bias.shape[1],  // B
                (float*)ones.d_data->get(), 1,  // 1
                &alpha,
                (float*)y.d_data->get(), y.shape[1]);  // self
    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];
    ensure(dy.shape == y.shape);
    // dx = dy * w'
    return { dy.matmul(w, false, true) };
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
      ensure(CUDNN_STATUS_SUCCESS == cudnnDestroyFilterDescriptor(filterDesc));
    if (convDesc)
      ensure(CUDNN_STATUS_SUCCESS == cudnnDestroyConvolutionDescriptor(convDesc));
  }

  vector<int> get_output_shape() {
    if (w_krnl.count() < 1) {
      w_krnl = Tensor({kernel_size, kernel_size, input_shape[1], filters}, true);
      if (use_bias) {
        w_bias = Tensor({1, filters, 1, 1}, 0.0f);
      }
      cudnnCreateConvolutionDescriptor(&convDesc);
      cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
      // ensure(CUDNN_STATUS_SUCCESS == cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));

      cudnnCreateFilterDescriptor(&filterDesc);
      cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
          filters, input_shape[1], kernel_size, kernel_size);
    }
    int nn = input_shape[0], cc = filters;
    int hh = (input_shape[2] + padding + padding - kernel_size) / stride + 1,
        ww = (input_shape[3] + padding + padding - kernel_size) / stride + 1;
    return {nn, cc, hh, ww};
  }

  string to_string() const {
    return "Convolution";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    float alpha = 1.0f, beta = 0.0f;
    vector<int> output_shape = get_output_shape(), cu_shape(4);
    output_shape[0] = xs[0].shape[0];
    ensure(CUDNN_STATUS_SUCCESS == cudnnGetConvolution2dForwardOutputDim(convDesc, xs[0].dataTensor->get(), filterDesc,
        &cu_shape[0], &cu_shape[1], &cu_shape[2], &cu_shape[3]));
    die_if(output_shape != cu_shape, "Conv layer not matching: %s & %s.",
        Tensor::stringify_shape(output_shape).c_str(), Tensor::stringify_shape(cu_shape).c_str());

    Tensor y(output_shape);

    cudnnConvolutionFwdAlgo_t convalgo;
    ensure(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardAlgorithm(devices[currentDev].hCudnn, xs[0].dataTensor->get(), filterDesc, convDesc,
        y.dataTensor->get(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convalgo));

    size_t sizeInBytes;
    ensure(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardWorkspaceSize(devices[currentDev].hCudnn, xs[0].dataTensor->get(), filterDesc, convDesc,
        y.dataTensor->get(), convalgo, &sizeInBytes));

    Tensor::DeviceMemory workspace(sizeInBytes);

    ensure(CUDNN_STATUS_SUCCESS == cudnnConvolutionForward(devices[currentDev].hCudnn, &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(),
        filterDesc, (float*)w_krnl.d_data->get(), convDesc, convalgo, workspace.get(), sizeInBytes,
        &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    if (use_bias)
      ensure(CUDNN_STATUS_SUCCESS == cudnnAddTensor(devices[currentDev].hCudnn,
        &alpha, w_bias.dataTensor->get(), (float*)w_bias.d_data->get(), &alpha, y.dataTensor->get(), (float*)y.d_data->get()));
    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];
    float alpha = 1.0f, beta = 0.0f;
    ensure(y.shape[1] == filters);
    int n = x.shape[0], c = x.shape[1], h = x.shape[2], w = x.shape[3];

    cudnnConvolutionBwdDataAlgo_t dalgo;
    ensure(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardDataAlgorithm(
                devices[currentDev].hCudnn, filterDesc, y.dataTensor->get(), convDesc, x.dataTensor->get(),
                CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &dalgo));

    size_t sizeInBytes;
    ensure(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardDataWorkspaceSize(
                devices[currentDev].hCudnn, filterDesc, y.dataTensor->get(), convDesc, x.dataTensor->get(), 
                dalgo, &sizeInBytes));

    Tensor::DeviceMemory workspace(sizeInBytes);

    Tensor dx({n, c, h, w});
    ensure(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardData(devices[currentDev].hCudnn, &alpha,
                filterDesc, (float*)w_krnl.d_data->get(),
                dy.dataTensor->get(), (float*)dy.d_data->get(),
                convDesc, dalgo, workspace.get(), sizeInBytes, &beta,
                dx.dataTensor->get(), (float*)dx.d_data->get()));
    return { move(dx) };
  }

  vector<Tensor> get_gradients(const Tensor &dy) const {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    vector<Tensor> grads = { Tensor(w_krnl.shape) };

    cudnnConvolutionBwdFilterAlgo_t falgo;
    ensure(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardFilterAlgorithm(
                devices[currentDev].hCudnn, x.dataTensor->get(), y.dataTensor->get(), convDesc, filterDesc,
                CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &falgo));

    size_t sizeInBytes;
    ensure(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionBackwardFilterWorkspaceSize(
                devices[currentDev].hCudnn, x.dataTensor->get(), y.dataTensor->get(), convDesc, filterDesc, 
                falgo, &sizeInBytes));

    Tensor::DeviceMemory workspace(sizeInBytes);

    float alpha = 1.0f, beta = 0.0f;
    ensure(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardFilter(devices[currentDev].hCudnn, &alpha,
                x.dataTensor->get(), (float*)x.d_data->get(),
                dy.dataTensor->get(), (float*)dy.d_data->get(),
                convDesc, falgo, workspace.get(), sizeInBytes, &beta,
                filterDesc, (float*)grads[0].d_data->get()));

    if (use_bias) {
      grads.push_back(Tensor(w_bias.shape));

      ensure(CUDNN_STATUS_SUCCESS == cudnnConvolutionBackwardBias(devices[currentDev].hCudnn, &alpha,
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


class BatchNormalization: public Layer {
  unsigned long steps;
  Tensor bnScal, bnBias, e_vari, e_mean, s_vari, s_mean, g_scal, g_bias;
  cudnnBatchNormMode_t bnMode;

public:
  BatchNormalization(): bnMode(CUDNN_BATCHNORM_SPATIAL /*CUDNN_BATCHNORM_PER_ACTIVATION*/), steps(1) {
  }

  ~BatchNormalization() {
  }

  vector<int> get_output_shape() {
    die_if(input_shape.size() != 4, "Currently Pooling Layer only suport 4D tensor (NCHW).");
    if (bnScal.count() < 1) {
      bnScal = Tensor({1, input_shape[1], 1, 1}, 1.0f);
      bnBias = Tensor({1, input_shape[1], 1, 1}, 0.0f);
      e_vari = Tensor({1, input_shape[1], 1, 1}, 1.0f);
      e_mean = Tensor({1, input_shape[1], 1, 1}, 0.0f);
      s_vari = Tensor({1, input_shape[1], 1, 1}, 0.0f);
      s_mean = Tensor({1, input_shape[1], 1, 1}, 0.0f);
      g_scal = Tensor({1, input_shape[1], 1, 1}, 0.0f);
      g_bias = Tensor({1, input_shape[1], 1, 1}, 0.0f);

      e_vari.trainable = e_mean.trainable = false;
    }
    return input_shape;
  }

  string to_string() const {
    return "BatchNormalization";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    // y = (x - e_mean) / sqrt(e_vari) * bnScale + bnBias
    die_if(false, "batchnorm not supported yet.");
    Tensor y(xs[0].shape);
    float alpha = 1.0f, beta = 0.0f;
    ensure(CUDNN_STATUS_SUCCESS == cudnnBatchNormalizationForwardTraining(devices[currentDev].hCudnn, bnMode,
      &alpha, &beta, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), y.dataTensor->get(), (float*)y.d_data->get(),
      bnScal.dataTensor->get(), (float*)bnScal.d_data->get(), (float*)bnBias.d_data->get(),
      1.0 / steps, // extra for training
      (float*)e_mean.d_data->get(), (float*)e_vari.d_data->get(), CUDNN_BN_MIN_EPSILON,
      (float*)s_mean.d_data->get(), (float*)s_vari.d_data->get())); // extra for training

    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx(dy.shape);
    float alpha = 1.0f, beta = 0.0f;

    ensure(CUDNN_STATUS_SUCCESS == cudnnBatchNormalizationBackward(devices[currentDev].hCudnn, bnMode,
      &alpha, &beta, &alpha, &beta, x.dataTensor->get(), (float*)x.d_data->get(), y.dataTensor->get(), (float*)y.d_data->get(),
      dx.dataTensor->get(), (float*)dx.d_data->get(), bnScal.dataTensor->get(),
      (float*)bnScal.d_data->get(), (float*)g_scal.d_data->get(), (float*)g_bias.d_data->get(), CUDNN_BN_MIN_EPSILON,
      (float*)s_mean.d_data->get(), (float*)s_vari.d_data->get()));

    ++steps;
    return { move(dx) };
  }

  vector<Tensor> get_weights() const {
    return {bnScal, bnBias, e_vari, e_mean};
  }

  vector<Tensor> get_gradients(const Tensor &dy) const {
    return {g_scal, g_bias, {}, {}};
  }
};

#endif
