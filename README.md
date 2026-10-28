# ai training power microcycling (in progress!!)

**Question:** what shapes the training power waveform, and which decisions reduce grid-relevant swings at the lowest throughput/energy cost?

**Rationale:** data centers are popping up pretty much everywhere nowadays, with big bets on ai infra. there's a lot of push for BTM energy, but most of these data centers 
still rely on the grid for power. the problem with AI power demand is that it's very spike-y and requires huge influxes and drops, which a traditional grid is not built to
handle. there are a lot of solutions in development to address this, with innovations for better supercapacitors or BESS systems to buffer microcycles.

**Purpose:** as someone who's passionate about both infra and model architectures, i wanted to understand if there were also software/architectural decisions we could make 
to mitigate microcycling and alleviate pressure on the grid + communities relying on those grids. the first step is to actually map different model decisions to their 
respective energy usage patterns. this repository's purpose is to serve as an experiment that measures those differences exactly.

## running it
one h100 sxm vm (~4-5 gpu hrs), analysis on my mac. edit code on the mac and re-run the step 1 rsync.

**1. copy code over** (mac)
```
rsync -av --exclude runs --exclude data --exclude __pycache__ ./ ubuntu@<ip>:ai_energy_usage/
ssh ubuntu@<ip>
cd ai_energy_usage
```

**2. install + data** (vm, mamba-ssm compile takes 20-40 min)
```
pip install -r requirements.txt && pip install --no-build-isolation mamba-ssm causal-conv1d
python data.py
```

**3. calibrate** (vm): `python calibration.py`, then set `nyquist_hz` and `nvml_window_s` at the top of `metrics.py` from its output

**4. smoke test** (vm): run `python train.py <arch> 0` for each of baseline, moe, moe_fine, mamba, hybrid until a few steps print, ctrl-c, then `rm -rf runs/*_0`

**5. pick batch_size** (in `train.py`): highest value where dense sits at ~85-90% of tdp and moe_fine still fits in memory

**6. sweep** (vm, ~3 hrs)
```
tmux new -s run    # ctrl-b d to detach, tmux attach -t run to come back
python sweep.py
```

**7. analyze** (mac)
```
rsync -av ubuntu@<ip>:ai_energy_usage/runs/ runs/
python metrics.py  # -> runs/summary.csv
```

**8. shut down the vm**
