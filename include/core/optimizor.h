
class Optimizor {

public:
  virtual void apply_updates(const vector<Tensor> &symbolic_gradients) = 0;
};


class MomentumOptimizor: public Optimizor {

  float momentum, lr, decay, k;
  vector<Tensor> symbolic_velocity, symbolic_weights;

public:
  MomentumOptimizor(const shared_ptr<Model> &model, float momentum = 0.9f, float lr = 0.01f, float decay = 0.0f):
      symbolic_weights(model->collect_all_weights()), momentum(momentum), lr(lr), k(0) {

    symbolic_velocity.resize(symbolic_weights.size());
    for (int i = 0; i < symbolic_weights.size(); ++i)
      symbolic_velocity[i] = Tensor(symbolic_weights[i].shape, 0.0f);
  }

  void apply_updates(const vector<Tensor> &symbolic_gradients) {
    die_if(symbolic_weights.size() != symbolic_gradients.size(), "The quantity of weights and gradients doesn't match.");

    float speed = lr;
    if (decay > 0)
      speed *= pow((1.0f + decay * k), -0.75f), ++k;

    for (int i = 0; i < symbolic_weights.size(); ++i) {
      if (!symbolic_weights[i].trainable)
        continue;
      symbolic_velocity[i].self_update(symbolic_gradients[i], speed / mpi_size, momentum / mpi_size);
      symbolic_velocity[i].allreduce();
      symbolic_weights[i].self_add(symbolic_velocity[i], -1.0f);
    }
  }
};


class SGDOptimizor: public Optimizor {

  float lr, decay, k;
  vector<Tensor> symbolic_weights;

public:
  SGDOptimizor(const shared_ptr<Model> &model, float lr = 0.01f, float decay = 0.0f):
      symbolic_weights(model->collect_all_weights()), lr(lr), decay(decay), k(0) {
  }

  void apply_updates(const vector<Tensor> &symbolic_gradients) {
    die_if(symbolic_weights.size() != symbolic_gradients.size(), "The quantity of weights and gradients doesn't match.");

    float speed = lr;
    if (decay > 0)
      speed *= pow((1.0f + decay * k), -0.75f), ++k;

    for (int i = 0; i < symbolic_weights.size(); ++i) {
      if (!symbolic_weights[i].trainable)
        continue;
      symbolic_gradients[i].allreduce();
      symbolic_weights[i].self_add(symbolic_gradients[i], -speed / mpi_size);
    }
  }
};
