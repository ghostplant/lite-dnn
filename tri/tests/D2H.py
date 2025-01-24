import torch
import triton, triton.language as tl
import os, sys
import time

try:
  import torch_maia
  import maia_athena
  port = int(os.environ.get('PORT', 0))
  torch_maia.load_firmware(port)
  maia_athena.get_nepal_device(port).set_global_hbm_limit(int(1024 * 1024 * 1024 * 40))
  DEVICE = torch.device(f'maia:{port}')
except:
  DEVICE = torch.device('cuda')

torch.manual_seed(0)

size = 1024 * 1024 * int(os.environ.get('MB', 32))
if size == 0:
    size = 4
y = torch.rand(size // 4, device='cpu').to(DEVICE)

is_cuda = (DEVICE.type == 'cuda')

while True:
  steps = 50
  t0 = time.perf_counter()
  for i in range(steps):
    z = y.to('cpu', non_blocking=is_cuda)
  if is_cuda:
    torch.cuda.synchronize()
  else:
    z.view(-1)[0].item()
  t1 = time.perf_counter()
  cost_us = (t1 - t0) / steps * 1e6
  print('DtoH', cost_us)
