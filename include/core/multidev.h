
class DeviceEvents {

public:
  vector<vector<CUevent>> freed, busy;
  unordered_map<CUevent, pair<function<void(const void*)>, const void*>> callbacks;

  DeviceEvents() {
  }

  ~DeviceEvents() {
    for (int c = 0; c < busy.size(); ++c) {
      Tensor::activateCurrentDevice(c);
      while (busy[c].size())
        recycle();
      for (auto event: freed[c])
        ensure(CUDA_SUCCESS == cuEventDestroy(event));
    }
  }

  CUevent recordEvent(int dev) {
    int backupDev = currentDev;
    if (freed.size() <= dev)
      freed.resize(dev + 1), busy.resize(dev + 1);

    CUevent event;
    if (freed[dev].size()) {
      event = freed[dev].back();
      freed[dev].pop_back();
    } else {
      Tensor::activateCurrentDevice(dev);
      ensure(CUDA_SUCCESS == cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
      Tensor::activateCurrentDevice(backupDev);
    }
    busy[dev].push_back(event);
    return event;
  }

  void setDependency(int after, int before) {
    CUevent event = recordEvent(before);
    ensure(CUDA_SUCCESS == cuEventRecord(event, devices[before].hStream));
    ensure(CUDA_SUCCESS == cuStreamWaitEvent(devices[after].hStream, event, 0));
  }

  void setLooseCallback(const function<void(const void*)> &cb, const void *args) {
    CUevent event = recordEvent(currentDev);
    callbacks[event] = {cb, args};
  }

  void recycle() {
    int backupDev = currentDev;
    for (int c = 0; c < busy.size(); ++c) {
      Tensor::activateCurrentDevice(c);
      for (int i = 0; i < busy[c].size(); ++i) {
        CUresult res = cuEventQuery(busy[c][i]);
        if (res == CUDA_SUCCESS) {
          auto cbarg = callbacks.find(busy[c][i]);
          if (cbarg != callbacks.end()) {
            cbarg->second.first(cbarg->second.second);
            callbacks.erase(cbarg);
          }
          freed[c].push_back(busy[c][i]);
          busy[c][i--] = busy[c].back();
          busy[c].pop_back();
          continue;
        }
        die_if(res != CUDA_ERROR_NOT_READY, "Unexpected event result code: %d", res);
      }
    }
    Tensor::activateCurrentDevice(backupDev);
  }
};

