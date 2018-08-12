
namespace lite_dnn {

 namespace apps {

  namespace imagenet_resnet50v1 {

    shared_ptr<Model> create_model(const string &image_ph, const string &label_ph, const vector<int> &image_shape, int n_class) {
      die_if(image_shape.size() != 3 || image_shape[1] != 224 || image_shape[2] != 224,
        "Only 3D shape of (3, 224, 224) is supported for ImageNet_Resnet50.");

      auto top_layer = make_shared<InputLayer>(image_ph, image_shape[0], image_shape[1], image_shape[2])
        ->then(make_shared<Convolution>(64, 7, 2, 3))
        ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
        ->then(make_shared<Pooling>(3, 2, CUDNN_POOLING_MAX));

      auto bottleneck_block_v1 = [&](shared_ptr<Layer> &input_layer, int depth, int depth_bottleneck, int stride) {
        auto shortcut = (depth == input_layer->get_output_shape()[1]) ? (stride == 1 ? input_layer : input_layer->then(make_shared<Pooling>(1, 2, CUDNN_POOLING_MAX)))
                          : input_layer->then(make_shared<Convolution>(depth, 1, stride))->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
        auto output = input_layer
          ->then(make_shared<Convolution>(depth_bottleneck, 1, stride))->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
          ->then(make_shared<Convolution>(depth_bottleneck, 3, 1, 1))->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
          ->then(make_shared<Convolution>(depth, 1, 1));

        return make_shared<Concat>(output, shortcut)
          ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU));
      };

      vector<int> layer_counts = {3, 4, 6, 3};
      for (int i = 0; i < layer_counts[0]; ++i)
        top_layer = bottleneck_block_v1(top_layer, 256, 64, 1);
      for (int i = 0; i < layer_counts[1]; ++i)
        top_layer = bottleneck_block_v1(top_layer, 512, 128, i == 0 ? 2 : 1);
      for (int i = 0; i < layer_counts[2]; ++i)
        top_layer = bottleneck_block_v1(top_layer, 1024, 256, i == 0 ? 2 : 1);
      for (int i = 0; i < layer_counts[3]; ++i)
        top_layer = bottleneck_block_v1(top_layer, 2048, 512, i == 0 ? 2 : 1);

      auto top_shape = top_layer->get_output_shape();
      die_if(top_shape.size() < 4 || top_shape[2] != top_shape[3], "Not supporting weight != height.");

      top_layer = top_layer->then(make_shared<Pooling>(top_shape[2], 1, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING))
        ->then(make_shared<Flatten>())
        ->then(make_shared<Dense>(n_class))
        ->then(make_shared<SoftmaxCrossEntropy>(label_ph));

      return top_layer->compile();
    }

  }
 }
}
