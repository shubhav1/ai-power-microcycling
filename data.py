import os
import numpy as np
import tiktoken
from datasets import load_dataset

# ~50M gpt-2 tokens of fineweb-edu. runs sample random windows, so a few minutes of training never runs out
n_tokens = 50_000_000

enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']
ds = load_dataset('HuggingFaceFW/fineweb-edu', name='sample-10BT', split='train', streaming=True)

os.makedirs('data', exist_ok=True)
tokens = np.empty(n_tokens, dtype=np.uint16)
n = 0
for doc in ds:
    toks = [eot] + enc.encode_ordinary(doc['text'])
    take = min(len(toks), n_tokens - n)
    tokens[n:n+take] = toks[:take]
    n += take
    if n == n_tokens:
        break
    if n % 5_000_000 < len(toks):
        print(f'{n:,} tokens')

tokens.tofile('data/fineweb_edu.bin')
print(f'wrote {n:,} tokens to data/fineweb_edu.bin')
