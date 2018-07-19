#ifndef __LITEDNN__MODEL__
#define __LITEDNN__MODEL__

float get_loss(const Tensor &data_loss) {
  vector<float> loss_data = data_loss.get_data();
  float loss = 0.0f;
  for (int i = 0; i < loss_data.size(); ++i) {
    float j = fabs(loss_data[i]);
    if (j >= 1e-8)
      loss += -j * log(j);
  }
  loss /= data_loss.shape[0];
  return loss;
}

float get_accuracy(const Tensor &data_pred, const Tensor &data_label) {
  assert(data_pred.shape.size() == 2 && data_pred.shape == data_label.shape);

  vector<float> real_data = data_label.get_data();
  vector<float> pred_data = data_pred.get_data();

  int tot = 0, acc = 0;
  for (int i = 0; i < data_pred.shape[0]; ++i) {
    int it = 0, jt = 0;
    for (int j = 1; j < data_pred.shape[1]; ++j) {
      if (pred_data[i * data_pred.shape[1] + it] < pred_data[i * data_pred.shape[1] + j])
        it = j;
      if (real_data[i * data_pred.shape[1] + jt] < real_data[i * data_pred.shape[1] + j])
        jt = j;
    }
    ++tot;
    if (it == jt)
      ++acc;
  }
  return acc * 100.0f / tot;
}

#endif
