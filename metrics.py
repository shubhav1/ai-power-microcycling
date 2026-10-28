import os
import json
import numpy as np
import pandas as pd
from scipy import signal

# these 2 vals will be replaced by calibrations.py's H100 output
nyquist_hz = 5.0 # nvml updates every ~100ms
nvml_window_s = 0.025 # each reading averages the last 25ms
warm_up_s = 60 # cut from the start of every run, same for all of them

def load_run(path):
    power = pd.read_csv(f'{path}/power.csv')
    marks = pd.read_csv(f'{path}/marks.csv')
    with open(f'{path}/meta.json') as f:
        meta = json.load(f)
    return power, marks, meta

def drop_repeats(power):
    """ keeps only real counter updates, since we collect updates more frequently than nvml actually does """
    updates = power[power.power_w.diff() != 0].copy()
    updates['t'] = updates.t - nvml_window_s / 2
    return updates

def step_starts(marks):
    return marks[(marks.phase == 'data') & (marks.step >= 0)].groupby('step').t_start.min()

def trim_warm_up(marks):
    """ returns (t_start, t_end), trimming the warm up and leaving only the data that can be analyzed.
        cuts the first warm_up_s seconds of training, anything after that (including mid-run hiccups) is kept """
    t_start = step_starts(marks).iloc[0] + warm_up_s # training starts at the first step
    t_end = marks[marks.phase == 'idle_post'].t_start.iloc[0] # and ends where the idle after it begins
    return t_start, t_end

def device_metrics(path):
    power, marks, meta = load_run(path)
    t0, t1 = trim_warm_up(marks)
    p = drop_repeats(power)
    p = p[(p.t >= t0) & (p.t <= t1)]
    tdp = meta['tdp_w']
    w = p.power_w.values

    # ramps only between real updates (>=~100ms apart), so these are lower bounds on the true ramp
    ramp = np.abs(np.diff(w) / np.diff(p.t.values)).max()

    # welch psd on a uniform resample, only up to the validated nyquist
    t_uniform = np.arange(t0, t1, 1 / (2 * nyquist_hz))
    w_uniform = np.interp(t_uniform, p.t.values, w)
    f, psd = signal.welch(w_uniform - w_uniform.mean(), fs=2 * nyquist_hz, nperseg=min(len(w_uniform), 1024))
    band = (f > 0.05) & (f <= nyquist_hz)

    starts = step_starts(marks)
    kept_starts = starts[(starts >= t0) & (starts <= t1)]
    return {
        'run': os.path.basename(path), 'condition': meta['condition'], 'rep': meta['rep'],
        'gpu_uuid': meta['uuid'],
        'mean_w': w.mean(),
        'p95_w': np.percentile(w, 95),
        'trough_w': np.percentile(w, 5),
        'swing_w': np.percentile(w, 95) - np.percentile(w, 5),
        'swing_frac': (np.percentile(w, 95) - np.percentile(w, 5)) / w.mean(),
        'swing_tdp': (np.percentile(w, 95) - np.percentile(w, 5)) / tdp,
        'ramp_w_s': ramp,
        'ramp_tdp_s': ramp / tdp, # %tdp/s maps straight onto pu/s at site scale
        'step_period_s': kept_starts.diff().median(),
        'dominant_hz': f[band][psd[band].argmax()],
        'mj_per_token': meta['energy_j'] / meta['tokens'] * 1000,
        'tok_s': meta['tokens'] / meta['train_seconds'],
        'mfu': meta['mfu'],
        'mean_temp_c': power[(power.t >= t0) & (power.t <= t1)].temp_c.mean(),
        'analyzed_s': t1 - t0,
    }

def average_step(path, bins=50):
    """ the power curve of one typical training step. nvml only gets a few readings per step, so every reading is
        placed by how far through its step it was (0 to 1), and readings are averaged across thousands of steps
        in `bins` slices. still blurred by nvml's own averaging window """
    power, marks, meta = load_run(path)
    t0, t1 = trim_warm_up(marks)
    p = drop_repeats(power)
    p = p[(p.t >= t0) & (p.t <= t1)]
    starts = step_starts(marks).values
    i = np.searchsorted(starts, p.t.values) - 1
    ok = (i >= 0) & (i < len(starts) - 1)
    i, t, w = i[ok], p.t.values[ok], p.power_w.values[ok]
    phase = (t - starts[i]) / (starts[i + 1] - starts[i])
    b = np.minimum((phase * bins).astype(int), bins - 1)
    avg_step = pd.DataFrame({'bin': b, 'power_w': w}).groupby('bin').power_w.agg(['mean', 'std', 'count'])
    avg_step['phase'] = (avg_step.index + 0.5) / bins
    if len(avg_step) < bins:
        # a step period that's a multiple of nvml's 100ms update always samples the same phases
        print(f'warning: {path}: only {len(avg_step)}/{bins} phase bins have readings')
    return avg_step

def squarewave_report(path='runs/squarewave'):
    """ how much of each known square wave nvml actually sees (calibrates to see accurate it is) """
    power = pd.read_csv(f'{path}/power.csv')
    marks = pd.read_csv(f'{path}/marks.csv')
    rows = []
    for period, m in marks.groupby('period_ms'):
        p = power[(power.t >= m.t_on.min() + 2) & (power.t <= m.t_off.max())] # skip the first 2s ramp-in
        w = p.power_w.values
        t_uniform = np.arange(p.t.min(), p.t.max(), 0.001) # 1 kHz, so even the 10ms (100 Hz) wave is representable
        w_uniform = np.interp(t_uniform, p.t.values, w)
        f, psd = signal.welch(w_uniform - w_uniform.mean(), fs=1000, nperseg=min(len(w_uniform), 16384))
        rows.append({'period_ms': period, 'true_hz': 1000 / period, 'seen_hz': f[1:][psd[1:].argmax()],
                     'swing_w': np.percentile(w, 95) - np.percentile(w, 5)})
    df = pd.DataFrame(rows)
    df['attenuation'] = df.swing_w / df.swing_w.iloc[-1] # relative to the slowest (2s) wave
    return df

if __name__ == '__main__':
    # usage: python metrics.py -> one row per run in runs/summary.csv
    runs = sorted(d for d in os.listdir('runs') if os.path.exists(f'runs/{d}/meta.json'))
    summary = pd.DataFrame([device_metrics(f'runs/{r}') for r in runs])
    summary.to_csv('runs/summary.csv', index=False)
    print(f'{len(summary)} runs -> runs/summary.csv')
