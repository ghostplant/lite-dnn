#!/usr/bin/env python

import os, sys, math
import torch
import pathlib, json
import time

try:
    from huggingface_hub import snapshot_download
    from safetensors.torch import safe_open
    from transformers import AutoTokenizer
except:
    print(f'Failed to import huggingface, please install the client with: {sys.executable} -m pip install "huggingface_hub[cli]" "transformers" "safetensors"')
    exit(0)

try:
  import autort
except:
  pass

try:
  if 'DEVICE' in os.environ:
    raise
  import torch_maia
  import maia_athena

  port = int(os.environ.get('PORT', 0))
  torch_maia.load_firmware(port)
  maia_athena.get_nepal_device(port).set_global_hbm_limit(int(1024 * 1024 * 1024 * 63))
  device = torch.device(f'maia:{port}')
except:
  device = os.environ.get('DEVICE', 'cpu')

os.environ['D3D12_ENABLE_FP16'] = '1'
model_id = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'
# model_id = 'Qwen/Qwen2.5-7B-Instruct'

snapshot_download(repo_id=model_id, local_dir=f"./{model_id}")

def load_to(filename, params):
  with safe_open(f'./{model_id}/{filename}.safetensors', framework='pt') as f:
    for k in f.keys():
      param[k] = f.get_tensor(k)
  return param

try:
  param = {}
  for i in range(8): load_to(f'model-{i+1:05d}-of-000008', param)
  # for i in range(4): load_to(f'model-{i+1:05d}-of-00004', param)
except FileNotFoundError:
  raise Exception('File not found')

tokenizer = AutoTokenizer.from_pretrained(f'./{model_id}')
config = json.loads(pathlib.Path(f'./{model_id}/config.json').read_text())

def print_keys(param):
  for k in param:
    print(f"{k}, {param[k].shape}")

for n_layers in range(1024):
  try:
    q, k, v = param[f'model.layers.{n_layers}.self_attn.q_proj.weight'], param[f'model.layers.{n_layers}.self_attn.k_proj.weight'], param[f'model.layers.{n_layers}.self_attn.v_proj.weight']
    vqk = torch.cat([v, q, k])
    del q, k, v, param[f'model.layers.{n_layers}.self_attn.q_proj.weight'], param[f'model.layers.{n_layers}.self_attn.k_proj.weight'], param[f'model.layers.{n_layers}.self_attn.v_proj.weight']
    param[f'model.layers.{n_layers}.self_attn.vqk_proj.weight'] = vqk

    q, k, v = param[f'model.layers.{n_layers}.self_attn.q_proj.bias'], param[f'model.layers.{n_layers}.self_attn.k_proj.bias'], param[f'model.layers.{n_layers}.self_attn.v_proj.bias']
    vqk_bias = torch.cat([v, q, k])
    del q, k, v, param[f'model.layers.{n_layers}.self_attn.q_proj.bias'], param[f'model.layers.{n_layers}.self_attn.k_proj.bias'], param[f'model.layers.{n_layers}.self_attn.v_proj.bias']
    param[f'model.layers.{n_layers}.self_attn.vqk_proj.bias'] = vqk_bias
  except KeyError:
    break

offload_before_layer = 63

for k in param:
  try:
    param[k] = param[k].to(torch.bfloat16)
    if not k.startswith('model.layers.') or int(k.split('.')[2]) < offload_before_layer:
      param[k] = param[k].to(device)
    print(f'Loading weight: {k} to {param[k].device}')
  except RuntimeError:
    raise Exception('Out of device memory, please try `export VMEM=1` before running this application again.')

# n_layers = offload_before_layer

rms_end_w = param['model.norm.weight']
weight_classify = param['lm_head.weight']
token_embedding_table = param['model.embed_tokens.weight']
data_type = token_embedding_table.dtype

rms_att_w = [param[f'model.layers.{i}.input_layernorm.weight'] for i in range(n_layers)]
rms_ffn_w = [param[f'model.layers.{i}.post_attention_layernorm.weight'] for i in range(n_layers)]
weight_o = [param[f'model.layers.{i}.self_attn.o_proj.weight'] for i in range(n_layers)]
weight_f1 = [param[f'model.layers.{i}.mlp.gate_proj.weight'] for i in range(n_layers)]
weight_f2 = [param[f'model.layers.{i}.mlp.down_proj.weight'] for i in range(n_layers)]
weight_f3 = [param[f'model.layers.{i}.mlp.up_proj.weight'] for i in range(n_layers)]
weight_vqk = [param[f'model.layers.{i}.self_attn.vqk_proj.weight'] for i in range(n_layers)]
bias_vqk = [param[f'model.layers.{i}.self_attn.vqk_proj.bias'] for i in range(n_layers)]

seq_len = 1024
n_heads = config['num_attention_heads']
hidden_size = token_embedding_table.size(-1)
head_size = hidden_size // n_heads # 128, dimension of each head
kv_heads = config['num_key_value_heads']
kv_dim = kv_heads * head_size
kv_groups = hidden_size // kv_dim
rope_theta = config['rope_theta']

assert n_heads % kv_heads == 0 # assure that GQA can be implemented by repeating kv_cache

token_embedding_table = token_embedding_table.view([token_embedding_table.size(0), n_heads, head_size])

key_cache_cpu = torch.zeros([n_layers, seq_len, kv_heads * head_size], dtype=data_type)
val_cache_cpu = torch.zeros([n_layers, seq_len, kv_heads * head_size], dtype=data_type)
key_cache_gpu = key_cache_cpu.to(device)
val_cache_gpu = val_cache_cpu.to(device)

att_f = torch.tensor([1 / math.sqrt(head_size)], dtype=torch.float32).to(device)

inv_freq_cpu = (1.0 / (rope_theta ** (torch.arange(0, head_size, 2).float() / head_size)).to(data_type))
inv_freq_cpu = torch.cat([inv_freq_cpu, inv_freq_cpu]).view(head_size)
inv_freq_gpu = inv_freq_cpu.to(device)

def rmsnorm(x, weight):
  return torch.nn.functional.rms_norm(x.float(), [x.size(-1)], weight.float(), 1e-6).view(-1).to(x.dtype)

def rotate_half(x):
  x1 = x[..., : x.shape[-1] // 2]
  x2 = x[..., x.shape[-1] // 2 :]
  return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, inv_freq, pos, sq_out, sk_out):
  inv = inv_freq * pos
  q_out = rotate_half(q) * torch.sin(inv) + q * torch.cos(inv)
  k_out = rotate_half(k) * torch.sin(inv) + k * torch.cos(inv)
  sq_out.copy_(q_out)
  sk_out.copy_(k_out)
  return sq_out

def eval(fn,  ctx=''):
  import time
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
  print(f'Operator ({ctx}), cost_us = {sorted(costs)[len(costs) // 2]:.2f}')

def forward(token, pos):
  x = token_embedding_table.select(0, token).view(1, hidden_size)

  for l in range(n_layers):
    is_cpu = rms_att_w[l].is_cpu
    key_cache = key_cache_cpu if is_cpu else key_cache_gpu
    val_cache = val_cache_cpu if is_cpu else val_cache_gpu
    x = x.to(rms_att_w[l].device)

    xb = rmsnorm(x, rms_att_w[l]).view(1, -1)
    local_cache = torch.matmul(xb, weight_vqk[l].t()) + bias_vqk[l]
    sv, sq, sk = torch.split(local_cache.view(-1), [kv_dim, hidden_size, kv_dim])
    val_cache[l].select(0, pos).copy_(sv)

    sq_out = torch.empty_like(sq).view(n_heads, head_size)
    sk_out = key_cache.select(0, l).narrow(0, pos, 1).view(kv_heads, head_size)

    apply_rotary_pos_emb(sq.view(n_heads, -1), sk.view(kv_heads, -1), inv_freq_gpu if not is_cpu else inv_freq_cpu, pos, sq_out, sk_out)

    b_sq = sq_out.view(n_heads, head_size)
    b_sk = key_cache.select(0, l).view(seq_len, kv_heads, head_size).narrow(0, 0, pos + 1)
    b_sv = val_cache.select(0, l).view(seq_len, kv_heads, head_size).narrow(0, 0, pos + 1)
    b_sk = b_sk.repeat_interleave(kv_groups, dim=1)
    b_sv = b_sv.repeat_interleave(kv_groups, dim=1)

    def pack_seq(x):
      x = x.view(-1, n_heads, head_size)
      y = torch.zeros([(x.size(0) + 31) // 32 * 32, n_heads, head_size], dtype=x.dtype, device=x.device)
      y[:x.size(0)].copy_(x)
      return y.permute(1, 0, 2).contiguous()

    if not is_cpu:
      original_seq = b_sk.size(0)
      xb = torch.matmul(b_sq.view(n_heads, 1, head_size), pack_seq(b_sk).permute(0, 2, 1)) / math.sqrt(b_sq.size(-1))
      xb = torch.nn.functional.softmax(xb[:, :, :original_seq].contiguous(), dim=-1)
      xb = torch.matmul(xb, b_sv.permute(1, 0, 2)).permute(1, 0, 2).contiguous()[:1]
    else:
      xb = torch.nn.functional.scaled_dot_product_attention(b_sq.view(-1, n_heads, head_size).permute(1, 0, 2).contiguous(), b_sk.view(-1, n_heads, head_size).permute(1, 0, 2).contiguous(), b_sv.view(-1, n_heads, head_size).permute(1, 0, 2).contiguous()).to(x.device).permute(1, 0, 2).contiguous().view(n_heads, head_size)

    xb = torch.matmul(xb.view(1, hidden_size), weight_o[l].t())
    x = x + xb
    xb = rmsnorm(x, rms_ffn_w[l])

    xb = torch.nn.functional.silu(torch.matmul(xb, weight_f1[l].t())) * torch.matmul(xb, weight_f3[l].t())
    xb = torch.matmul(xb, weight_f2[l].t())
    x = x + xb

  x = rmsnorm(x.to(rms_end_w.device), rms_end_w)
  logits = torch.matmul(x, weight_classify.t())
  return logits


if __name__ == '__main__':
  prompt = 'Calculate the result of 1 / (sqrt(3) + sqrt(5))' if len(sys.argv) <= 1 else ' '.join(sys.argv[1:])

  messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
  ]
  text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
  )

  prompt_tokens = tokenizer([text], return_tensors="pt")['input_ids'][0].to(device)

  with torch.no_grad():
    pos, token = 0, prompt_tokens[0]

    print()
    while pos < seq_len - 2:
      decoded_word = tokenizer.decode(token)
      sys.stdout.write(decoded_word)
      if decoded_word[:1] == '<' and decoded_word[-1:] == '>':
        sys.stdout.write('\n')
      sys.stdout.flush()
      # print(f'\n{time.perf_counter()}')
      logits = forward(token, pos).cpu()
      if pos < prompt_tokens.shape[0] - 1:
        next_token = prompt_tokens[pos + 1]
      else:
        next_token = torch.argmax(logits.cpu(), dim=-1)
      if next_token == tokenizer.eos_token_id and pos > prompt_tokens.shape[0]:
        break
      token = next_token
      pos += 1
  print()
