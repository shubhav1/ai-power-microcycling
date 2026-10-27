# each condition is a set of overrides on the baseline globals at the top of train.py.
# architecture at matched compute: same tokens/step, seq len, optimizer, precision, ~124M active params

conditions = {
    'baseline': {}, # dense gpt-2 small
    'moe': {'model_type': 'moe'}, # coarse: 8 experts, top-2
    'moe_fine': {'model_type': 'moe_fine'}, # fine-grained: 64 experts, top-7 + 1 shared
    'mamba': {'model_type': 'mamba'},
    'hybrid': {'model_type': 'hybrid'}, # mamba-2 with attention every 8th layer
}
