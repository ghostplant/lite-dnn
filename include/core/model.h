
class Model {

public:
  shared_ptr<Layer> top_layer;

  vector<Layer*> layers;
  unordered_map<const Layer*, int> index;

  Model(const shared_ptr<Layer> &top_layer): top_layer(top_layer), index({{top_layer.get(), 1}}) {
    queue<Layer*> que;
    que.push(top_layer.get());
    int indice = 1;

    while (que.size()) {
      auto layer = que.front(); que.pop();
      layers.push_back(layer);

      for (auto &parent: layer->parents) {
        if (!index.count(parent.get())) {
          index[parent.get()] = ++indice;
          que.push(parent.get());
        }
      }
    }
    for (auto &it: index)
      it.second = indice - it.second;

    sort(layers.begin(), layers.end(), [&](const Layer *x, const Layer *y) {
      if (x->depth != y->depth)
        return x->depth < y->depth;
      return index[x] < index[y];
    });
  }

  void summary() {
    size_t parameter_count = 0;
    putchar('\n');
    for (auto &layer: layers) {
      printf(" => layer-#%02d D(%d): %20s, output_shape: %15s, from:", index[layer], layer->depth, layer->to_string().c_str(),
        Tensor::stringify_shape(layer->get_output_shape(), 1).c_str());
      if (layer->parents.size() == 0)
        printf(" (none)");
      for (auto &parent: layer->parents)
        printf(" #%02d", index[parent.get()]);
      putchar('\n');
      for (auto &weight: layer->get_weights())
        parameter_count += weight.count();
    }

    string params = to_string(parameter_count), decor;
    for (int i = params.size() % 3, prev = 0; i <= params.size(); prev = i, i += 3)
      decor += (decor.size() ? "," : "") + params.substr(prev, i - prev);
    printf("\n  [@] Total Parameters: %s\n\n", decor.c_str());
  }

  bool load_weights_from_file(const char *weight_path) {
    FILE *fp = fopen(weight_path, "rb");
    bool success = (fp != nullptr);

    if (fp) {
      unsigned version;
      if (sizeof(version) != fread(&version, 1, sizeof(version), fp) || version != 1)
        success = false;
    }

    for (auto &layer: layers) {
      if (!success)
        break;
      for (auto &weight: layer->get_weights()) {
        vector<float> host(weight.count());
        if (host.size() != fread(host.data(), sizeof(float), host.size(), fp)) {
          success = false;
          break;
        }
        weight.set_data(host.data());
      }
    }

    if (success) {
      ssize_t offset = ftell(fp);
      fseek(fp, 0, SEEK_END);
      if (ftell(fp) != offset)
        success = false;
    }
    if (fp != nullptr)
      fclose(fp);
    printf("  [@] Loading weights data: %s.\n\n", success ? "YES" : "NO");
  }

  bool save_weights_to_file(const char *weight_path) {
    FILE *fp = fopen(weight_path, "wb");
    bool success = (fp != nullptr);

    unsigned version = 0x1;
    if (sizeof(version) != fwrite(&version, 1, sizeof(version), fp))
      success = false;

    for (auto &layer: layers) {
      if (!success)
        break;
      for (auto &weight: layer->get_weights()) {
        auto host = weight.get_data();
        if (host.size() != fwrite(host.data(), sizeof(float), host.size(), fp)) {
          success = false;
          break;
        }
      }
    }

    if (fp != nullptr)
      fclose(fp);
    printf("  [@] Saving weights data: %s.\n\n", success ? "YES" : "NO");
  }

  Tensor predict(const unordered_map<string, Tensor> &feed_dict = {}) {
    unordered_map<Layer*, Tensor> ys;

    for (auto &layer: layers) {
      vector<Tensor> xs;
      for (auto it: layer->parents) {
        ensure(ys.count(it.get()) > 0);
        xs.push_back(ys[it.get()]);
      }
      ensure(ys.count(layer) == 0);
      ys[layer] = layer->forward(xs, feed_dict);
    }
    return ys[top_layer.get()];
  }

  vector<Tensor> collect_all_gradients(const unordered_map<string, Tensor> &feed_dict) {
    unordered_map<Layer*, Tensor> dys = {{top_layer.get(), {}}};

    vector<Tensor> grads;
    for (int i = layers.size() - 1; i >= 0; --i) {
      auto &layer = layers[i];
      ensure(dys.count(layer) > 0);

      auto dxs = layer->backward(dys[layer], feed_dict);
      die_if(dxs.size() != layer->parents.size(), "the size of loss vector doesn't match the number of parent nodes.");

      auto curr = layer->get_gradients(dys[layer]);
      grads.insert(grads.end(), curr.begin(), curr.end());

      for (int i = layer->parents.size() - 1; i >= 0; --i) {
        auto parent = layer->parents[i].get();

        if (dys.count(parent) == 0) {
          dys[parent] = dxs[i];
        } else {
          dys[parent].self_add(dxs[i]);
        }
      }
    }
    return move(grads);
  }

  vector<Tensor> collect_all_weights() {
    vector<Tensor> weights;
    for (int i = layers.size() - 1; i >= 0; --i) {
      auto &layer = layers[i];

      auto curr = layer->get_weights();
      if (curr.size() > 0)
        weights.insert(weights.end(), curr.begin(), curr.end());
    }
    return move(weights);
  }
};

