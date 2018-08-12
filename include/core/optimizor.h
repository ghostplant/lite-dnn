
class Optimizor {

public:
  virtual void apply_updates(vector<Tensor> &symbolic_weights, const vector<Tensor> &symbolic_gradients) = 0;
};


class MomentumOptimizor {

  float momentum, lr, decay, k;
  vector<Tensor> symbolic_velocity;

public:
  MomentumOptimizor(float momentum = 0.9f, float lr = 0.01f, float decay = 0.0f): momentum(momentum), lr(lr), k(0) {
  }

  void apply_updates(vector<Tensor> &symbolic_weights, const vector<Tensor> &symbolic_gradients) {
    die_if(symbolic_weights.size() != symbolic_gradients.size(), "The quantity of weights and gradients doesn't match.");

    if (symbolic_velocity.size() != symbolic_weights.size()) {
      symbolic_velocity.resize(symbolic_weights.size());
      for (int i = 0; i < symbolic_weights.size(); ++i)
        symbolic_velocity[i] = Tensor(symbolic_weights[i].shape, 0.0f);
    }

    float speed = lr;
    if (decay > 0)
      speed *= pow((1.0f + decay * k), -0.75f), ++k;

    for (int i = 0; i < symbolic_weights.size(); ++i) {
      symbolic_velocity[i].self_update(symbolic_gradients[i], speed, momentum);
      symbolic_weights[i].self_add(symbolic_velocity[i], -1.0f);
    }
  }
};


class SGDOptimizor {

  float lr, decay, k;

public:
  SGDOptimizor(float lr = 0.01f, float decay = 0.0f): lr(lr), decay(decay), k(0) {
  }

  void apply_updates(vector<Tensor> &symbolic_weights, const vector<Tensor> &symbolic_gradients) {
    die_if(symbolic_weights.size() != symbolic_gradients.size(), "The quantity of weights and gradients doesn't match.");

    float speed = lr;
    if (decay > 0)
      speed *= pow((1.0f + decay * k), -0.75f), ++k;

    for (int i = 0; i < symbolic_weights.size(); ++i)
      symbolic_weights[i].self_add(symbolic_gradients[i], -speed);
  }
};
