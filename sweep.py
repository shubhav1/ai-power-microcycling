import os
import sys
import random
import shutil
import subprocess
from conditions import conditions

# usage: python sweep.py [reps]
# everything runs on the same gpu in one go, in random order, with a dense baseline every 5 runs so i can see if the gpu drifts
# vm setup in the readme
# all terminal output also goes to runs/sweep.log, and the software versions + gpu details to runs/env.txt and runs/gpu.txt

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5 # need at least 5, with 3 mann-whitney can't get under p = 0.05 no matter what

runs = [(c, r) for c in conditions if c != 'baseline' for r in range(reps)]
random.shuffle(runs)
order = []
b = 0
for i, run in enumerate(runs):
    if i % 5 == 0:
        order.append(('baseline', b))
        b += 1
    order.append(run)
order.append(('baseline', b))

os.makedirs('runs', exist_ok=True)
log = open('runs/sweep.log', 'a')

def say(line):
    """ print to the terminal and save to runs/sweep.log """
    print(line, flush=True)
    log.write(line + '\n')
    log.flush()

def train(cond, rep):
    """ runs train.py, showing its output live and saving every line of it to runs/sweep.log """
    p = subprocess.Popen([sys.executable, 'train.py', cond, str(rep)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, env={**os.environ, 'PYTHONUNBUFFERED': '1'}) # unbuffered so lines show up as they happen
    for line in p.stdout:
        say(line.rstrip('\n'))
    if p.wait() != 0:
        say(f'{cond} rep {rep} crashed, stopping the sweep')
        sys.exit(1)

# record exactly what software + gpu this sweep ran on
with open('runs/env.txt', 'w') as f:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=f)
with open('runs/gpu.txt', 'w') as f:
    subprocess.run(['nvidia-smi', '-q'], stdout=f)

say(f'{len(order)} runs + 1 pre-warm, in this order:')
for i, (cond, rep) in enumerate(order):
    say(f'  {i+1}. {cond} rep {rep}')

train('baseline', 999) # warm up run
shutil.rmtree('runs/baseline_999') # deletes warm up run

# runs all conditions
for i, (cond, rep) in enumerate(order):
    say(f'[{i+1}/{len(order)}] {cond} rep {rep}')
    train(cond, rep)
