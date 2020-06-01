import os

start = 32
n_repetitions = 60
log_parent = "./runs/clean/full/random5/{}/"

for i in range(start, n_repetitions):
    log_dir = log_parent.format(i)
    call_str = "python full_system.py --parameters ./Parameters/parameters_full.json --logdir {}".format(log_dir)
    os.system(call_str)
