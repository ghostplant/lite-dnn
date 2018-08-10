#ifndef __LITEDNN_LAYERS__
#define __LITEDNN_LAYERS__


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

  Tensor predict(const unordered_map<string, Tensor> &feed_dict = {}) {
    vector<Tensor> xs;
    for (auto it: parents)
      xs.push_back(it->predict(feed_dict));
    return this->forward(xs, feed_dict);
  }

  vector<Tensor> collect_all_gradients(const unordered_map<string, Tensor> &feed_dict, const Tensor &dy = {}) {
    auto dxs = this->backward(dy, feed_dict);
    vector<Tensor> grads;
    die_if(dxs.size() != parents.size(), "the size of loss vector doesn't match the number of parent nodes.");
    for (int i = 0; i < parents.size(); ++i) {
      auto prev = parents[i]->collect_all_gradients(feed_dict, dxs[i]);
      grads.insert(grads.end(), prev.begin(), prev.end());
    }
    auto curr = this->get_gradients(dy);
    grads.insert(grads.end(), curr.begin(), curr.end());
    return move(grads);
  }

  vector<Tensor> collect_all_weights() {
    vector<Tensor> weights;
    for (auto it: parents) {
      auto prev = it->collect_all_weights();
      weights.insert(weights.end(), prev.begin(), prev.end());
    }
    auto curr = this->get_weights();
    weights.insert(weights.end(), curr.begin(), curr.end());
    return move(weights);
  }

  shared_ptr<Layer> then(const shared_ptr<Layer> &that) {
    assert(this->input_shape.size() > 0);
    that->parents = { shared_from_this() };
    that->input_shape = this->get_output_shape();
    return that;
  }

  shared_ptr<Layer> summary(bool top = true) {
    if (top) putchar('\n');

    for (int i = 0; i < this->parents.size(); ++i)
      this->parents[i]->summary(false);
    printf(" => layer: %20s, output_shape: %s\n", this->to_string().c_str(),
        Tensor::stringify_shape(this->get_output_shape(), 1).c_str());

    if (top) putchar('\n');
    return shared_from_this();
  }

  bool load_weights_from_file(const char *weight_path, FILE *fp = nullptr) {
    if (fp == nullptr) {
      fp = fopen(weight_path, "rb");
      if (fp == nullptr)
        return false;
      printf("  [@] Loading the weights file: ");
    }
    bool succ = true;
    try {
      for (int i = 0; i < this->parents.size(); ++i)
        if (!this->parents[i]->load_weights_from_file(nullptr, fp))
          throw fp;

      for (auto &weight: this->get_weights()) {
        vector<float> host(weight.count());
        if (host.size() != fread(host.data(), sizeof(float), host.size(), fp))
          throw fp;
        weight.set_data(host.data());
      }
    } catch (...) {
      succ = false;
    }

    if (weight_path != nullptr) {
      ssize_t offset = ftell(fp);
      fseek(fp, 0, SEEK_END);
      if (ftell(fp) != offset)
        succ = false;

      fclose(fp);
      puts(succ ? "YES.\n" : "NO.\n");
    }
    return succ;
  }

  bool save_weights_to_file(const char *weight_path, FILE *fp = nullptr) {
    if (fp == nullptr) {
      fp = fopen(weight_path, "wb");
      if (fp == nullptr)
        return false;
      printf("  [@] Saving the weights file: ");
    }
    bool succ = true;
    try {
      for (int i = 0; i < this->parents.size(); ++i)
        if (!this->parents[i]->save_weights_to_file(nullptr, fp))
          throw fp;

      for (auto &weight: this->get_weights()) {
        auto host = weight.get_data();
        if (host.size() != fwrite(host.data(), sizeof(float), host.size(), fp))
          throw fp;
      }
    } catch (...) {
      succ = false;
    }

    if (weight_path != nullptr) {
      fclose(fp);
      puts(succ ? "YES.\n" : "NO.\n");
    }
    return succ;
  }


  vector<shared_ptr<Layer>> parents;
  vector<Tensor> cacheTensors;
  vector<int> input_shape;
};


class InputLayer: public Layer {

public:
  string place_holder;

  InputLayer(const string &place_holder, int channel, int height = -1, int width = -1): place_holder(place_holder) {
    if (height > 0 && width > 0)
      input_shape = {-1, channel, height, width};
    else
      input_shape = {-1, channel};
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
    assert(CUDNN_STATUS_SUCCESS == cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
        &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));
    cacheTensors = {y};
    return y;
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &y = cacheTensors[0];
    auto it = feed_dict.find(place_holder);
    die_if(it == feed_dict.end(), "Cannot find item `%s` in feed_dict.", place_holder.c_str());
    const Tensor &_dy = it->second;

    assert(_dy.shape == y.shape);

    float posi = 1.0f / _dy.shape[0], nega = -1.0f / _dy.shape[0];
    size_t len = _dy.count();

    Tensor dx(_dy.shape, 0.0f);
    cublasSaxpy(cublasHandle, len, &posi, (float*)y.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    cublasSaxpy(cublasHandle, len, &nega, (float*)_dy.d_data->get(), 1, (float*)dx.d_data->get(), 1);
    return { move(dx) };
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

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    Tensor y(xs[0].shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationForward(cudnnHandle, activationDesc, \
        &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx = dy;
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnActivationBackward(cudnnHandle, activationDesc,
        &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
        x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
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

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    assert(xs[0].shape.size() == 4);

    Tensor y(xs[0].shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelForward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
        &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnLRNCrossChannelBackward(cudnnHandle, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
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

  vector<int> get_output_shape() {
    die_if(input_shape.size() != 4, "Currently Pooling Layer only suport 4D tensor.");
    return {input_shape[0], input_shape[1], (input_shape[2] - (size - stride)) / stride, (input_shape[3] - (size - stride)) / stride};
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    auto shape = this->get_output_shape();
    shape[0] = xs[0].shape[0];
    Tensor y(shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingForward(cudnnHandle, poolDesc,
        &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(), &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];

    Tensor dx(x.shape);
    float alpha = 1.0f, beta = 0.0f;
    assert(CUDNN_STATUS_SUCCESS == cudnnPoolingBackward(cudnnHandle, poolDesc,
        &alpha, y.dataTensor->get(), (float*)y.d_data->get(), dy.dataTensor->get(), (float*)dy.d_data->get(),
        x.dataTensor->get(), (float*)x.d_data->get(), &beta, dx.dataTensor->get(), (float*)dx.d_data->get()));
    return { move(dx) };
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

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    size_t _reversed_size;
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(xs[0].dataTensor->get(), &_reversed_size));

    if (reversed_size == ~0LU || _reversed_size > reversed_size) {
      reversed_size = _reversed_size;
      reversed = make_shared<DeviceMemory>(reversed_size);
    }

    Tensor y(xs[0].shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutForward(cudnnHandle, dropDesc, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(),
      y.dataTensor->get(), (float*)y.d_data->get(), reversed->get(), reversed_size));
    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];
    assert(CUDNN_STATUS_SUCCESS == cudnnRestoreDropoutDescriptor(dropDesc, cudnnHandle, drop_prob, states->get(), states_size, seed));

    size_t _reversed_size;
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutGetReserveSpaceSize(y.dataTensor->get(), &_reversed_size));
    assert(_reversed_size <= reversed_size);

    Tensor dx(x.shape);
    assert(CUDNN_STATUS_SUCCESS == cudnnDropoutBackward(cudnnHandle, dropDesc, dy.dataTensor->get(), (float*)dy.d_data->get(),
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
      assert(input_shape.size() == 2);
      w = Tensor({input_shape[1], channels}, true);
      bias = Tensor({1, channels}, 0.0f);
    }
    return {input_shape[0], channels};
  }

  string to_string() const {
    return "Dense";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    assert(xs[0].shape.size() == 2);

    auto y = xs[0].matmul(w, false, false);
    // y = xs[0] * w + ones * bias';
    assert(y.shape.size() == 2 && bias.shape.size() == 2 && y.shape[1] == bias.shape[1] && y.shape[0] <= ones.shape[0]);
    // auto wx_b = y.add(ones.reshape({xs[0].shape[0], 1}, true).matmul(bias, false, false));
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
    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
    const Tensor &x = cacheTensors[0], &y = cacheTensors[1];
    assert(dy.shape == y.shape);
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
      assert(CUDNN_STATUS_SUCCESS == cudnnDestroyFilterDescriptor(filterDesc));
    if (convDesc)
      assert(CUDNN_STATUS_SUCCESS == cudnnDestroyConvolutionDescriptor(convDesc));
  }

  vector<int> get_output_shape() {
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
    int nn = input_shape[0], cc = filters, hh = (input_shape[2] + padding + padding - max(0, kernel_size - stride)) / stride,
        ww = (input_shape[3] + padding + padding - max(0, kernel_size - stride)) / stride;
    return {nn, cc, hh, ww};
  }

  string to_string() const {
    return "Convolution";
  }

  Tensor forward(const vector<Tensor> &xs, const unordered_map<string, Tensor> &feed_dict) {
    float alpha = 1.0f, beta = 0.0f;
    vector<int> output_shape = get_output_shape(), cu_shape(4);
    output_shape[0] = xs[0].shape[0];
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolution2dForwardOutputDim(convDesc, xs[0].dataTensor->get(), filterDesc,
        &cu_shape[0], &cu_shape[1], &cu_shape[2], &cu_shape[3]));
    assert(output_shape == cu_shape);

    Tensor y(output_shape);

    cudnnConvolutionFwdAlgo_t convalgo;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardAlgorithm(cudnnHandle, xs[0].dataTensor->get(), filterDesc, convDesc,
        y.dataTensor->get(), CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &convalgo));

    size_t sizeInBytes;
    assert(CUDNN_STATUS_SUCCESS == cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, xs[0].dataTensor->get(), filterDesc, convDesc,
        y.dataTensor->get(), convalgo, &sizeInBytes));

    DeviceMemory workspace(sizeInBytes);

    assert(CUDNN_STATUS_SUCCESS == cudnnConvolutionForward(cudnnHandle, &alpha, xs[0].dataTensor->get(), (float*)xs[0].d_data->get(),
        filterDesc, (float*)w_krnl.d_data->get(), convDesc, convalgo, workspace.get(), sizeInBytes,
        &beta, y.dataTensor->get(), (float*)y.d_data->get()));

    if (use_bias)
      assert(CUDNN_STATUS_SUCCESS == cudnnAddTensor(cudnnHandle,
        &alpha, w_bias.dataTensor->get(), (float*)w_bias.d_data->get(), &alpha, y.dataTensor->get(), (float*)y.d_data->get()));
    cacheTensors = {xs[0], y};
    return move(y);
  }

  vector<Tensor> backward(const Tensor &dy, const unordered_map<string, Tensor> &feed_dict) {
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
    return { move(dx) };
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

#endif
