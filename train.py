import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from models import build, active_params, flops_per_token
from sampler import PowerSampler, gpu_info
from conditions import conditions

# usage: python train.py <condition> <rep>
cond = sys.argv[1]
rep = int(sys.argv[2])

# baseline hyperparameters (conditions.py overrides these)
model_type = 'dense' # dense, moe, moe_fine, mamba, hybrid (see models/)
batch_size = 16 # phase 0 raises this until steady power is >=85-90% of tdp
block_size = 1024
vocab_size = 50304 # gpt-2's 50257 padded to a multiple of 64
learning_rate = 6e-4
run_seconds = 360 # first ~1 min is warm up, wont be coutned in metrics
idle_seconds = 10 # idle before and after training, captures start/stop transients
peak_flops = {'H100 PCIe': 756e12, 'H100 NVL': 835e12, 'H100': 989e12, 'A100': 312e12} # dense bf16 tensor core peak, for mfu

globals().update(conditions[cond])

out_dir = f'runs/{cond}_{rep}'
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + rep)
device = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True

# data: random windows from one fineweb-edu token file (see data.py)
def get_batch():
    # memmap per call so the file isn't held in ram, same as nanoGPT
    data = np.memmap('data/fineweb_edu.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True) # copies batch to GPU asynchronously

# marks: (step, phase, t_start, t_end) on the same perf_counter clock as the power sampler
marks = []
def host_phase(step, phase, t0):
    marks.append((step, phase, t0, time.perf_counter()))

# gpu-side phase boundaries within a step; converted to wall time after the one sync per step
events = []
def record(phase):
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    events.append((phase, e))

info = gpu_info()
sampler = PowerSampler()
sampler.start()
t0 = time.perf_counter()
time.sleep(idle_seconds)
host_phase(-1, 'idle_pre', t0)

t0 = time.perf_counter()
model = build(model_type, vocab_size, block_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1, fused=True)
host_phase(-1, 'init', t0)
n_params = sum(p.numel() for p in model.parameters())
n_active = active_params(model)
flops_tok = flops_per_token(model, block_size)
peak = next((v for k, v in peak_flops.items() if k in info['name']), float('nan')) # first match, so 'H100 PCIe' wins over 'H100'
print(f'{cond} rep {rep}: {model_type}, {n_params/1e6:.1f}M params ({n_active/1e6:.1f}M active)')

energy_start = sampler.energy_j()
t_train = time.perf_counter()
step = 0
tokens = 0
while time.perf_counter() - t_train < run_seconds:
    events.clear()
    record('data')
    x, y = get_batch()
    record('fwd')
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(x, y)
    record('bwd')
    loss.backward()
    record('opt')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    record('end')

    loss_val = loss.item() # one sync per step
    t_end = time.perf_counter()
    end_event = events[-1][1]
    for (phase, e), (_, e_next) in zip(events[:-1], events[1:]):
        marks.append((step, phase, t_end - e.elapsed_time(end_event) / 1000, t_end - e_next.elapsed_time(end_event) / 1000))
    tokens += batch_size * block_size

    if step % 100 == 0:
        tok_s = tokens / (time.perf_counter() - t_train)
        print(f'step {step}: loss {loss_val:.4f}, {tok_s:,.0f} tok/s, mfu {tok_s * flops_tok / peak:.1%}, {sampler.last_power:.0f} W')
    step += 1

torch.cuda.synchronize()
t_stop = time.perf_counter()
energy_j = sampler.energy_j() - energy_start
mfu = tokens / (t_stop - t_train) * flops_tok / peak

t0 = time.perf_counter()
time.sleep(idle_seconds)
host_phase(step, 'idle_post', t0)
sampler.stop()

sampler.save(f'{out_dir}/power.csv')
pd.DataFrame(marks, columns=['step', 'phase', 't_start', 't_end']).to_csv(f'{out_dir}/marks.csv', index=False)
meta = {
    'condition': cond, 'rep': rep, 'overrides': conditions[cond], 'model_type': model_type,
    'params': n_params, 'active_params': n_active,
    'batch_size': batch_size, 'block_size': block_size,
    'steps': step, 'tokens': tokens, 'train_seconds': t_stop - t_train, 'energy_j': energy_j,
    'flops_per_token': flops_tok, 'mfu': mfu,
    'torch': torch.__version__, 'cuda': torch.version.cuda, **info,
}
with open(f'{out_dir}/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(f'done: {step} steps, {tokens / (t_stop - t_train):,.0f} tok/s, mfu {mfu:.1%}, {energy_j / tokens * 1000:.3f} mJ/token')
