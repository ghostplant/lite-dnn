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

@triton.jit
def sin_kernel(
    x_ptr,  # *Pointer* to first input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
    ST: tl.constexpr,
):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + block_start + offsets)
    tl.store(output_ptr + block_start + offsets, tl.sin(x))

ST = 1

def sin(x: torch.Tensor, output: torch.Tensor, BLOCK_SIZE: int):
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, BLOCK_SIZE * ST),)
    sin_kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, ST=ST)
    return output

def eval(fn, BLOCK_SIZE):
  costs = []
  for i in range(10):
    steps = 10
    t0 = time.perf_counter()
    for i in range(steps):
      z = fn()
    z.view(-1)[0].item()
    t1 = time.perf_counter()
    cost_us = (t1 - t0) / steps * 1e6
    costs += [cost_us]
  print(f'Operator (BLOCK_SIZE={BLOCK_SIZE}), cost_us = {sorted(costs)[len(costs) // 2]:.2f}')


torch.manual_seed(0)
size = 1024 * 1024 * int(os.environ.get('MB', 128))
x = torch.rand(size).to(DEVICE)
output = torch.zeros_like(x)
output_torch = torch.sin(x).cpu()
output_triton = sin(x, output, 1024).cpu()
print(output_torch)
print(output_triton)

diff = torch.max(torch.abs(output_torch - output_triton))
print(
    f"The maximum difference between torch and triton is "
    f"{diff}"
)
assert diff < 1e-3

eval(lambda: torch.sin(x),  0)
for i in [1024, 4096, 1024 * 16, 1024 * 64]:
  eval(lambda: sin(x, output, i), i)
