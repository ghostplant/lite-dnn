
bool is_standard_name(const string &name) {
  for (auto c: name)
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\'' || c == '"')
      return false;
  return true;
}

auto load_images(const string &images, string cache_path = "/tmp/dataset") {
  // images: cifar10 / mnist / catsdogs
  if (cache_path.back() != '/')
    cache_path += "/";
  cache_path += images + '/';
  die_if(!is_standard_name(cache_path) || !is_standard_name(images), "Illegal chars exist in `cache_path` or `images`.");

  make_dirs(cache_path);
  FILE *fp = fopen((cache_path + ".success").c_str(), "rb");
  if (!fp) {
    printf("Downloading source dataset to '%s'..\n", cache_path.c_str());
    die_if(0 != system((string() + "curl -L 'https://github.com/ghostplant/lite-dnn/releases/download/lite-dataset/images-" + images + ".tar.gz' | tar xzvf - -C '" + cache_path + "' >/dev/null").c_str()),
        "Failed to download dataset.");
    die_if((fp = fopen((cache_path + ".success").c_str(), "wb")) == nullptr, "No access to complete saving dataset.");
  }
  fclose(fp);
  return pair<string, string>({cache_path + "train", cache_path + "validate"});
}
