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

M, K = int(os.environ.get('M', 11008)), int(os.environ.get('K', 4096))

def test(batch):
  x = torch.rand([batch, K], dtype=torch.bfloat16, device='cpu').to(DEVICE)
  y = torch.rand([M, K], dtype=torch.bfloat16, device='cpu').to(DEVICE)

  costs = []
  for i in range(10):
    steps = 10
    t0 = time.perf_counter()
    for i in range(steps):
      z = torch.matmul(x,  y.t())
    z.view(-1)[0].item()
    t1 = time.perf_counter()
    cost_s = (t1 - t0) / steps
    costs += [cost_s]
  sc = sorted(costs)[len(costs) // 2]
  print(f'Operator (batch, M, K={batch}, {M}, {K}), cost_s = {sc * 1e6:.2f}, mem = {(x.numel() + y.numel() + batch * M) * 1e-9 / sc:.2f} GB/s, perf = {(x.numel() * M * 2) * 1e-12 / sc:.4f} TFlops')

for s in [1, 4, 4096, 1024 * 16]:
  test(s)
