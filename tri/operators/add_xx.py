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

def test(size_mb):
  size = 1024 * 1024 * size_mb
  x = torch.rand(size // 4, device='cpu').to(DEVICE)

  costs = []
  for i in range(10):
    steps = 10
    t0 = time.perf_counter()
    for i in range(steps):
      z = torch.add(x, x)
    z.view(-1)[0].item()
    t1 = time.perf_counter()
    cost_us = (t1 - t0) / steps * 1e6 * 2 / 3
    costs += [cost_us]
  print(f'Operator (size_mb={size_mb}), cost_us = {sorted(costs)[len(costs) // 2]:.2f}')

for s in [1, 8, 32, 128, 512]:
  test(s)
