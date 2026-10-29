import os
import pandas as pd
import matplotlib.pyplot as plt
from metrics import load_run, trim_warm_up, average_step

# usage (from the repo root): python analysis/plot_steps.py -> analysis/step_power.png
# one subplot per architecture showing gpu power across one average training step (after the warm up).
# thin lines are each rep's average step, the thick line is the mean of the reps.
# nvml averages every reading over ~25ms, so anything sharper than that gets smoothed out

archs = {
    'baseline': 'dense',
    'moe': 'coarse moe (8 experts, top-2)',
    'moe_fine': 'fine moe (64 experts, top-7 + 1 shared)',
    'mamba': 'mamba-2',
    'hybrid': 'hybrid (mamba-2 + attention every 8th layer)',
}
color = '#2a6fdb'
tdp_w = 700

def phase_ms(path):
    """ average ms per step spent in each phase (data, fwd, bwd, opt), after the warm up """
    power, marks, meta = load_run(path)
    t0, t1 = trim_warm_up(marks)
    m = marks[(marks.step >= 0) & (marks.t_start >= t0) & (marks.t_end <= t1)]
    return ((m.t_end - m.t_start) * 1000).groupby([m.step, m.phase]).sum().unstack().mean()

fig, axes = plt.subplots(len(archs), 1, figsize=(8, 11), sharex=True, sharey=True)
for ax, (arch, name) in zip(axes, archs.items()):
    runs = sorted(d for d in os.listdir('runs') if d.rsplit('_', 1)[0] == arch and os.path.exists(f'runs/{d}/meta.json'))
    steps = [average_step(f'runs/{r}') for r in runs]
    phases = pd.DataFrame([phase_ms(f'runs/{r}') for r in runs]).mean()
    step_ms = phases[['data', 'fwd', 'bwd', 'opt']].sum()

    # shade fwd and opt so the phases are visible, label any wide enough to fit text
    start = phases['data']
    for phase, shade in [('fwd', '#eeeeee'), ('bwd', None), ('opt', '#dddddd')]:
        end = start + phases[phase]
        if shade:
            ax.axvspan(start, end, color=shade, lw=0)
        if phases[phase] > 15:
            ax.text((start + end) / 2, tdp_w + 25, phase, ha='center', va='bottom', fontsize=8, color='#555555')
        start = end

    for s in steps:
        ax.plot(s.phase * step_ms, s['mean'], color=color, lw=1, alpha=0.35)
    mean = pd.concat([s['mean'] for s in steps], axis=1).mean(axis=1)
    ax.plot(steps[0].phase * step_ms, mean, color=color, lw=2)

    ax.axhline(tdp_w, color='#888888', lw=1, ls='--')
    ax.set_title(f'{name}  ·  {step_ms:.0f} ms/step', loc='left', fontsize=10)
    ax.set_ylabel('power (W)')
    ax.grid(axis='y', color='#eeeeee', lw=0.8)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].text(axes[0].get_xlim()[1], tdp_w + 5, 'tdp 700 W', ha='right', va='bottom', fontsize=8, color='#888888')
axes[0].set_ylim(top=tdp_w + 70)
axes[-1].set_xlabel('time into the training step (ms)')
fig.suptitle('gpu power across one average training step, h100 sxm', x=0.01, ha='left', fontsize=12)
fig.text(0.01, 0.005, '5 reps per architecture, 60 s warm up cut. nvml averages each reading over ~25 ms, so sharp edges are smoothed.',
         fontsize=7, color='#666666')
fig.tight_layout(rect=(0, 0.015, 1, 1))
fig.savefig('analysis/step_power.png', dpi=200)
print('-> analysis/step_power.png')
