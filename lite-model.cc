/*
  DNN Trainging based on CUBLAS/CUDNN/NCCL/OpenMPI

  Maintainer: Wei CUI <ghostplant@qq.com>

  Benchmark on Nvida Tesla P100:

  ---------------------------------------------------------------------------------
       Model            | batch_size  |    Keras + TF_CUDA    |  Lite-DNN (C++14)
  ---------------------------------------------------------------------------------
     mnist_mlp          |    32       |    8.34 sec/epoll     |  1.03 sec/epoll
     mnist_cnn          |    128      |    3.24 sec/epoll     |  1.35 sec/epoll
     cifar10_lenet      |    128      |    2.68 sec/epoll     |  1.15 sec/epoll
     imagenet_resnet50  |    64       |    149.10 images/sec  |  243.22 images/sec
  ---------------------------------------------------------------------------------
*/


#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>

#include <core/tensor.h>
#include <core/layers.h>
#include <core/model.h>
#include <core/optimizor.h>

#include <core/generator.h>
#include <core/dataset.h>

// #include <apps/resnet50.h>
#include <apps/alexnet.h>

static inline unsigned long get_microseconds() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000000LU + tv.tv_usec;
}


int main(int argc, char **argv) {
  Tensor::init();

  /* 
  auto model = make_shared<InputLayer>("image_place_0", gen->channel, gen->height, gen->width)
    ->then(make_shared<Flatten>())
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(512))
    ->then(make_shared<Activation>(CUDNN_ACTIVATION_RELU))
    ->then(make_shared<Dense>(gen->n_class))
    ->then(make_shared<SoftmaxCrossEntropy>("label_place_0"))
    ->compile(); */

  int batch_size = 64, steps = 50000;

  printf("Creating generator for GPU-%d ..\n", mpi_rank);
  auto dataset = load_images("flowers");
  auto gen = make_shared<ImageDataGenerator>(224, 224, batch_size, dataset.first, 4, true);
  auto val_gen = make_shared<ImageDataGenerator>(224, 224, batch_size, dataset.second, 1, false);

  auto model = lite_dnn::apps::imagenet_alexnet::create_model(
    "image_place_0", "label_place_0", {gen->channel, gen->height, gen->width}, gen->n_class);
  auto optimizor = make_shared<MomentumOptimizor>(model, 0.9f, 0.001f, 0.001f);
  // auto optimizor = make_shared<SGDOptimizor>(model, 0.001f, 0.001f);

  if (mpi_rank == 0)
    model->summary();

  model->load_weights_from_file("weights.lw");
  vector<Tensor> weights = model->collect_all_weights();
  Tensor::synchronizeCurrentDevice();

  long lastClock = get_microseconds(), last_k = 0;

  for (int k = 1; k <= steps; ++k) {

    gen->recycleBuffer();
    auto batch_data = gen->next_batch();

    auto feed_dict = unordered_map<string, Tensor>({{"image_place_0", batch_data[0]}, {"label_place_0", batch_data[1]}});
    auto predicts = model->predict(feed_dict);

    auto grad = model->collect_all_gradients(feed_dict);

    optimizor->apply_updates(grad);

    long metric_frequency = 50, save_frequency = 1000;
    ensure(save_frequency % metric_frequency == 0);

    if (k % metric_frequency == 0 || k == 1) {
      auto lacc = predicts.compute_loss_and_accuracy(batch_data[1]);

      val_gen->recycleBuffer();
      auto val_batch_data = val_gen->next_batch();
      auto val_predicts = model->predict({{"image_place_0", val_batch_data[0]}});
      auto val_lacc = val_predicts.compute_loss_and_accuracy(val_batch_data[1]);
      double during = (get_microseconds() - lastClock) * 1e-6f;
      printf("==> [GPU-%d] step = %d (%.2lf images/s): loss = %.4f, top1 = %.2f%%, top5 = %.2f%%, v_loss = %.4f, v_top1 = %.2f%%, v_top5 = %.2f%%, during = %.2fs\n",
        mpi_rank, k, (k - last_k) * batch_size / during, lacc["loss"], lacc["top_1_acc"], lacc["top_5_acc"], val_lacc["loss"], val_lacc["top_1_acc"], val_lacc["top_5_acc"], during);

      if (k % save_frequency == 0 || k == steps) {
        if (mpi_rank == 0) {
          printf("Saving model weights ..\n");
          model->save_weights_to_file("weights.lw");
          Tensor::synchronizeCurrentDevice();
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      lastClock = get_microseconds(), last_k = k;
    }
  }
  printf("==> [GPU-%d] Training finished.\n", mpi_rank);
  Tensor::quit();
  return 0;
}
