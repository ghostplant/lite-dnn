
namespace lite_dnn {

 namespace apps {

  namespace imagenet_alexnet {

    shared_ptr<Model> create_model(const string &image_ph, const string &label_ph, const vector<int> &image_shape, int n_class) {
      die_if(image_shape.size() != 3 || image_shape[1] != 224 || image_shape[2] != 224,
        "Only 3D shape of (3, 224, 224) is supported for ImageNet_Alexnet.");

      auto top_layer = CreateLayer(InputLayer({.input = image_ph, .channel = 3, .height = image_shape[1], .width = image_shape[2]}))
        ->then(CreateLayer(Convolution({.filters = 96, .kernel_size = 11, .stride = 4, .padding = 2})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        // ->then(CreateLayer(LRN({.depthRadius = 4, .bias = 1.0, .lrnAlpha = 0.001 / 9.0, .lrnBeta = 0.75})))
        ->then(CreateLayer(Pooling({.size = 3, .stride = 2, .mode = CUDNN_POOLING_MAX})))
        ->then(CreateLayer(Convolution({.filters = 256, .kernel_size = 5, .stride = 1, .padding = 2})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        // ->then(CreateLayer(LRN({.depthRadius = 4, .bias = 1.0, .lrnAlpha = 0.001 / 9.0, .lrnBeta = 0.75})))
        ->then(CreateLayer(Pooling({.size = 3, .stride = 2, .mode = CUDNN_POOLING_MAX})))
        ->then(CreateLayer(Convolution({.filters = 384, .kernel_size = 3, .stride = 1, .padding = 1})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        // ->then(CreateLayer(LRN({.depthRadius = 4, .bias = 1.0, .lrnAlpha = 0.001 / 9.0, .lrnBeta = 0.75})))
        ->then(CreateLayer(Convolution({.filters = 256, .kernel_size = 3, .stride = 1, .padding = 1})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        // ->then(CreateLayer(LRN({.depthRadius = 4, .bias = 1.0, .lrnAlpha = 0.001 / 9.0, .lrnBeta = 0.75})))
        ->then(CreateLayer(Pooling({.size = 3, .stride = 2, .mode = CUDNN_POOLING_MAX})))
        ->then(CreateLayer(Flatten({})))
        ->then(CreateLayer(Dense({.channels = 4096})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        // ->then(CreateLayer(Dropout({.drop_prob = 0.25})))
        ->then(CreateLayer(Dense({.channels = 4096})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        // ->then(CreateLayer(Dropout({.drop_prob = 0.25})))
        ->then(CreateLayer(Dense({.channels = n_class})))
        ->then(CreateLayer(SoftmaxCrossEntropy({.label = label_ph})));

      return top_layer->compile();
    }
  }

  namespace cifar10_alexnet {

    shared_ptr<Model> create_model(const string &image_ph, const string &label_ph, const vector<int> &image_shape, int n_class) {
      die_if(image_shape.size() != 3 || image_shape[1] != 32 || image_shape[2] != 32,
        "Only 3D shape of (3, 32, 32) is supported for Cifar10_Alexnet.");

      auto top_layer = CreateLayer(InputLayer({.input = image_ph, .channel = 3, .height = image_shape[1], .width = image_shape[2]}))
        ->then(CreateLayer(Convolution({.filters = 64, .kernel_size = 5})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        ->then(CreateLayer(Pooling({.size = 3, .stride = 2, .mode = CUDNN_POOLING_MAX})))
        ->then(CreateLayer(LRN({.depthRadius = 4, .bias = 1.0, .lrnAlpha = 0.001 / 9.0, .lrnBeta = 0.75})))
        ->then(CreateLayer(Convolution({.filters = 64, .kernel_size = 5})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        ->then(CreateLayer(LRN({.depthRadius = 4, .bias = 1.0, .lrnAlpha = 0.001 / 9.0, .lrnBeta = 0.75})))
        ->then(CreateLayer(Pooling({.size = 3, .stride = 2, .mode = CUDNN_POOLING_MAX})))
        ->then(CreateLayer(Flatten({})))
        ->then(CreateLayer(Dense({.channels = 384})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        ->then(CreateLayer(Dense({.channels = 192})))
        ->then(CreateLayer(Activation({.mode = CUDNN_ACTIVATION_RELU})))
        ->then(CreateLayer(Dense({.channels = n_class})))
        ->then(CreateLayer(SoftmaxCrossEntropy({.label = label_ph})));

      return top_layer->compile();
    }
  }

 }
}
