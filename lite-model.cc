/*
  mnist_mlp based on CUBLAS/CUDNN

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

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn_v7.h>
#include <nccl.h>
#include <mpi.h>

#include <core/tensor.h>
#include <core/layers.h>
#include <core/model.h>
#include <core/optimizor.h>

#include <core/generator.h>
#include <core/dataset.h>

#include <apps/resnet50.h>
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
  auto dataset = load_images("cifar10");
  auto gen = make_shared<ImageDataGenerator>(dataset.first, 224, 224, 4, batch_size);
  auto val_gen = make_shared<ImageDataGenerator>(dataset.second, 224, 224, 1, batch_size);

  auto model_replias = lite_dnn::apps::imagenet_resnet50v1::create_model("image_place_0", "label_place_0", {gen->channel, gen->height, gen->width}, gen->n_class);
  auto optimizor = make_shared<MomentumOptimizor>(model_replias, 0.9f, 0.001f / mpi_size, 0.001f);

  if (mpi_rank == 0)
    model_replias->summary();

  model_replias->load_weights_from_file("weights.lw");
  vector<Tensor> weights = model_replias->collect_all_weights();
  Tensor::synchronizeCurrentDevice();

  long lastClock = get_microseconds(), last_k = 0;

  for (int k = 1; k <= steps; ++k) {

    gen->recycleBuffer();
    auto batch_data = gen->next_batch(batch_size);

    auto feed_dict = unordered_map<string, Tensor>({{"image_place_0", batch_data[0]}, {"label_place_0", batch_data[1]}});
    auto predicts = model_replias->predict(feed_dict);

    auto grad = model_replias->collect_all_gradients(feed_dict);
    for (int j = 0; j < grad.size(); ++j)
      ensure(0 == ncclAllReduce((const void*)grad[j].d_data->get(), (void*)grad[j].d_data->get(), grad[j].count(), ncclFloat, ncclSum, comm, devices[currentDev].hStream));

    optimizor->apply_updates(grad);

    long total_batch = batch_size * mpi_size, metric_frequency = 50, save_frequency = 1000;
    ensure(save_frequency % metric_frequency == 0);

    if (k % metric_frequency == 0 || k == 1) {
      auto lacc = predicts.get_loss_and_accuracy_with(batch_data[1]);

      val_gen->recycleBuffer();
      auto val_batch_data = val_gen->next_batch(batch_size);
      auto val_predicts = model_replias->predict({{"image_place_0", val_batch_data[0]}});
      auto val_lacc = val_predicts.get_loss_and_accuracy_with(val_batch_data[1]);
      double during = (get_microseconds() - lastClock) * 1e-6f;
      printf("==> [GPU-%d] step = %d (batch = %ld; %.2lf images/sec): loss = %.4f, acc = %.2f%%, val_loss = %.4f, val_acc = %.2f%%, during = %.2fs\n",
        mpi_rank, k, total_batch, (k - last_k) * total_batch / during, lacc.first, lacc.second, val_lacc.first, val_lacc.second, during);

      if (k % save_frequency == 0 || k == steps) {
        if (mpi_rank == 0) {
          printf("Saving model weights ..\n");
          model_replias->save_weights_to_file("weights.lw");
          Tensor::synchronizeCurrentDevice();
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
      lastClock = get_microseconds(), last_k = k;
    }
  }

  Tensor::quit();
  return 0;
}
