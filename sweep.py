import sys
import random
import shutil
import subprocess
from conditions import conditions

# usage: python sweep.py [reps]
# everything runs on the same gpu in one go, in random order, with a dense baseline every 5 runs so i can see if the gpu drifts
# vm setup in the readme

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

print(f'{len(order)} runs + 1 pre-warm')

subprocess.run(['python', 'train.py', 'baseline', '999'], check=True) # warm up run
shutil.rmtree('runs/baseline_999') # deletes warm up run

# runs all conditions
for i, (cond, rep) in enumerate(order):
    print(f'[{i+1}/{len(order)}] {cond} rep {rep}')
    subprocess.run(['python', 'train.py', cond, str(rep)], check=True)
