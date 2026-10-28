import os
import time
import pandas as pd
import torch
from sampler import PowerSampler, gpu_info
from metrics import squarewave_report

# resolution pilot: a 50% duty matmul square wave with known on/off times.
# comparing what nvml sees against the truth sets which frequencies later results can claim,
# and measures how often h100's instant power field actually updates (nobody's published it)
periods_ms = [10, 20, 50, 100, 130, 200, 500, 1000, 2000] # 130 isn't a multiple of any round update interval, so it shows aliasing
seconds_each = 20
rest_seconds = 5
poll_s = 0.001 # 10x faster than training's sampler, so updates down to ~1ms show up
n = 4096 # ~0.2ms per bf16 matmul on an H100, fine-grained enough to hit 5ms half-periods

print(gpu_info()) # also confirms nvml can read this gpu at all
out_dir = 'runs/squarewave'
os.makedirs(out_dir, exist_ok=True)
a = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)
b = torch.randn(n, n, device='cuda', dtype=torch.bfloat16)

sampler = PowerSampler(interval=poll_s)
sampler.start()
marks = []
for period in periods_ms:
    half = period / 2000
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < seconds_each:
        t_on = time.perf_counter()
        while time.perf_counter() - t_on < half:
            a @ b
            torch.cuda.synchronize()
        t_off = time.perf_counter()
        marks.append((period, t_on, t_off))
        time.sleep(max(0.0, half - (time.perf_counter() - t_off)))
    print(f'{period} ms done, last power {sampler.last_power:.0f} W')
    time.sleep(rest_seconds)
sampler.stop()

sampler.save(f'{out_dir}/power.csv')
pd.DataFrame(marks, columns=['period_ms', 't_on', 't_off']).to_csv(f'{out_dir}/marks.csv', index=False)
print(squarewave_report(out_dir).to_string(index=False))

# how often the reading actually changes = nvml's real update interval on this gpu
power = pd.read_csv(f'{out_dir}/power.csv')
update_s = power.t[power.power_w.diff() != 0].diff().median()
print(f'nvml updates every ~{update_s * 1000:.1f} ms -> set nyquist_hz = {1 / (2 * update_s):.1f} in metrics.py')
if update_s < 0.02:
    print('updates are faster than 20ms: lower PowerSampler interval in sampler.py so training polls at least 2x faster than this')
