import torch
import triton
import triton.language as tl
import os, time
torch.manual_seed(0)

try:
  import torch_maia
  import maia_athena

  port = int(os.environ.get('PORT', 0))
  torch_maia.load_firmware(port)
  maia_athena.get_nepal_device(port).set_global_hbm_limit(int(1024 * 1024 * 1024 * 4))
  DEVICE = torch.device(f'maia:{port}')
except:
  DEVICE = torch.device('cuda')

@triton.jit
def triton_kernel(
    x_ptr,
    y_ptr,
    N: tl.constexpr,
    M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid[:, None] * M + tl.arange(0, M)[None, :]
    x_val = tl.load(x_ptr + offsets)
    x_sum = tl.sum(x_val, axis=1)
    tl.store(y_ptr + (pid + tl.arange(0, 1)), x_sum)


def triton_fn(x):
    output = torch.empty([x.size(0)], device=x.device, dtype=x.dtype)
    triton_kernel[(x.size(0),)](x, output, x.size(0), x.size(1))
    return output

def eval(fn):
  costs = []
  for i in range(10):
    steps = 100
    t0 = time.perf_counter()
    for i in range(steps):
      z = fn()
    z.view(-1)[0].item()
    t1 = time.perf_counter()
    cost_us = (t1 - t0) / steps * 1e6
    costs += [cost_us]
  print(f'Operator, cost_us = {sorted(costs)[len(costs) // 2]:.2f}')


torch.manual_seed(0)
x = torch.rand([4096, 4096], device=DEVICE)
output_torch = torch.sum(x, dim=1)
output_triton = triton_fn(x)
print(output_torch.cpu())
print(output_triton.cpu())
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

eval(lambda: torch.sum(x, dim=1))
eval(lambda: triton_fn(x))

if DEVICE.type == 'cuda':
    import autort
    try:
      fn = autort.export(ir='reduce_sum_f32[N] +=! a[N, M]', inputs=[f'a=float32[N:{x.size(0)},M:{x.size(1)}]'], config='tune:50')
      autort.perform(lambda: fn(x), text='AutoRT')
      eval(lambda: fn(x))
    except:
      pass
