#ifndef __LITEDNN_DATASET__
#define __LITEDNN_DATASET__


static pair<vector<int>, vector<float>> ReadNormalDataset(const char* dataset) {
  auto read_uint32 = [&](FILE *fp) {
    uint32_t val;
    assert(fread(&val, sizeof(val), 1, fp) == 1);
    return __builtin_bswap32(val);
  };

  const int UBYTE_MAGIC = 0x800;
  FILE *fp;
  if ((fp = fopen(dataset, "rb")) == NULL) {
    fprintf(stderr, "Cannot open file: %s\n", dataset);
    exit(1);
  }

  uint32_t header, length;
  header = read_uint32(fp);
  length = read_uint32(fp);
  header -= UBYTE_MAGIC;

  assert(header >= 1 && header <= 4);
  if (header == 1) { // output_shape = (N, max(val) + 1),  max(val) <= 255
    vector<uint8_t> raw(length);
    assert(fread(raw.data(), 1, raw.size(), fp) == raw.size());

    uint32_t width = 0;
    for (int i = 0; i < raw.size(); ++i)
      width = max(width, (uint32_t)raw[i]);
    ++width;

    vector<int> shape = {(int)length, (int)width};
    vector<float> tensor(length * width);
    for (int i = 0; i < length; ++i)
      tensor[i * width + raw[i]] = 1.0f;
    return {move(shape), move(tensor)};

  } else if (header == 2) { // shape = (N, C),  may support max(val) > 255
    assert(0); // unsupported

  } else if (header == 3) { // shape = (N, 1, H, W)
    uint32_t h = read_uint32(fp);
    uint32_t w = read_uint32(fp);
    uint32_t width = h * w;

    vector<int> shape = {(int)length, 1, (int)h, (int)w};
    vector<float> tensor(length * width);
    vector<uint8_t> raw(width);
    for (int i = 0; i < length; ++i) {
      assert(fread(raw.data(), 1, raw.size(), fp) == raw.size());
      for (int j = 0; j < width; ++j)
        tensor[i * width + j] = raw[j] / 255.0f;
    }
    return {move(shape), move(tensor)};

  } else if (header == 4) { // shape = (N, C, H, W)
    uint32_t c = read_uint32(fp);
    uint32_t h = read_uint32(fp);
    uint32_t w = read_uint32(fp);
    uint32_t width = c * h * w;

    vector<int> shape = {(int)length, (int)c, (int)h, (int)w};
    vector<float> tensor(length * width);
    vector<uint8_t> raw(width);
    for (int i = 0; i < length; ++i) {
      assert(fread(raw.data(), 1, raw.size(), fp) == raw.size());
      for (int j = 0; j < width; ++j)
        tensor[i * width + j] = raw[j] / 255.0f;
    }
    return {move(shape), move(tensor)};

  }
  assert(0);
  return {{}, {}};
}

#endif
