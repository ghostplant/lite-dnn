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

y = torch.randn([1], dtype=torch.float32, device='cpu').to(DEVICE)

while True:
  steps = 100000
  t0 = time.perf_counter()
  for i in range(steps):
    z = torch.log(y)
  z.view(-1)[0].item()
  t1 = time.perf_counter()
  cost_us = (t1 - t0) / steps * 1e6
  print('Launch', cost_us)
