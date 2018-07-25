#ifndef __LITEDNN__MODEL__
#define __LITEDNN__MODEL__

pair<float, float> get_loss_and_accuracy(const Tensor &data_pred, const Tensor &data_label) {
  assert(data_pred.shape.size() == 2 && data_pred.shape == data_label.shape);

  vector<float> pred_data = data_pred.get_data();
  vector<float> real_data = data_label.get_data();

  float loss = 0.0f;
  for (int i = 0; i < pred_data.size(); ++i) {
    if (fabs(real_data[i]) >= 1e-8)
      loss += -real_data[i] * log(pred_data[i]);
  }
  loss /= pred_data.size();

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
  return {loss, acc * 100.0f / tot};
}

#endif
