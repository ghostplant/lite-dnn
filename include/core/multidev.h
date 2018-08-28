
class DeviceEvents {

public:
  vector<vector<CUevent>> freed, busy;

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

  void setDependency(int after, int before) {
    int backupDev = currentDev;
    int top = max(before, after) + 1;
    if (freed.size() < top)
      freed.resize(top), busy.resize(top);

    CUevent event;
    if (freed[before].size()) {
      event = freed[before].back();
      freed[before].pop_back();
    } else {
      Tensor::activateCurrentDevice(before);
      ensure(CUDA_SUCCESS == cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
      Tensor::activateCurrentDevice(backupDev);
    }
    busy[before].push_back(event);
    ensure(CUDA_SUCCESS == cuEventRecord(event, devices[before].hStream));
    ensure(CUDA_SUCCESS == cuStreamWaitEvent(devices[after].hStream, event, 0));
  }

  void recycle() {
    int backupDev = currentDev, ncnt = 0;
    for (int c = 0; c < busy.size(); ++c) {
      Tensor::activateCurrentDevice(c);
      for (int i = 0; i < busy[c].size(); ++i) {
        CUresult res = cuEventQuery(busy[c][i]);
        if (res == CUDA_SUCCESS) {
          freed[c].push_back(busy[c][i]);
          busy[c][i--] = busy[c].back();
          busy[c].pop_back();
          ++ncnt;
          continue;
        }
        die_if(res != CUDA_ERROR_NOT_READY, "Unexpected event result code: %d", res);
      }
    }
    Tensor::activateCurrentDevice(backupDev);
  }
};

