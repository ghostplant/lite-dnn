
namespace lite_dnn {

 namespace apps {

  namespace imagenet_alexnet {

    shared_ptr<Model> create_model(const string &image_ph, const string &label_ph, const vector<int> &image_shape, int n_class) {
      die_if(image_shape.size() != 3 || image_shape[1] != 224 || image_shape[2] != 224,
        "Only 3D shape of (3, 224, 224) is supported for ImageNet_Alexnet.");

      auto top_layer = make_shared<InputLayer>(image_ph, image_shape[0], image_shape[1], image_shape[2])
        ->then(make_shared<Convolution>(96, 11, 4, 2))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
        ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
        ->then(make_shared<Convolution>(256, 5, 1, 2))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
        ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
        ->then(make_shared<Convolution>(384, 3, 1, 1))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
        ->then(make_shared<Convolution>(256, 3, 1, 1))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
        ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
        ->then(make_shared<Flatten>())
        ->then(make_shared<Dense>(4096))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<Dropout>(0.25))
        ->then(make_shared<Dense>(4096))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<Dropout>(0.25))
        ->then(make_shared<Dense>(n_class))
        ->then(make_shared<SoftmaxCrossEntropy>(label_ph));

      return top_layer->compile();
    }
  }

  namespace cifar10_alexnet {

    shared_ptr<Model> create_model(const string &image_ph, const string &label_ph, const vector<int> &image_shape, int n_class) {
      die_if(image_shape.size() != 3 || image_shape[1] != 32 || image_shape[2] != 32,
        "Only 3D shape of (3, 32, 32) is supported for Cifar10_Alexnet.");

      auto top_layer = make_shared<InputLayer>(image_ph, image_shape[0], image_shape[1], image_shape[2])
        ->then(make_shared<Convolution>(64, 5, true))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
        ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
        ->then(make_shared<Convolution>(64, 5, true))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<LRN>(4, 1.0, 0.001 / 9.0, 0.75))
        ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX))
        ->then(make_shared<Flatten>())
        ->then(make_shared<Dense>(384))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<Dense>(192))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<Dense>(n_class))
        ->then(make_shared<SoftmaxCrossEntropy>(label_ph));

      return top_layer->compile();
    }
  }

 }
}
