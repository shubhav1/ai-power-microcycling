from .dense import GPT
from .moe import MoE

def build(model_type, vocab_size, block_size):
    """ every model takes (idx, targets) and returns (logits, loss), ~124M active params """
    if model_type == 'dense':
        return GPT(vocab_size, block_size)
    if model_type == 'moe':
        return GPT(vocab_size, block_size, ffwd=lambda C: MoE(C, n_exp=8, top_k=2))
    if model_type == 'moe_fine':
        return GPT(vocab_size, block_size, ffwd=lambda C: MoE(C, n_exp=64, top_k=7, n_shared=1))
    # imported here so dense/moe runs don't need mamba-ssm installed
    if model_type == 'mamba':
        from .mamba import Mamba
        return Mamba(vocab_size)
    if model_type == 'hybrid':
        from .hybrid import Hybrid
        return Hybrid(vocab_size)
    raise ValueError(f'unknown model_type {model_type}')

def active_params(model):
    """ params touched per token (moe counts top_k of n_exp routed experts) """
    total = sum(p.numel() for p in model.parameters())
    for m in model.modules():
        if isinstance(m, MoE):
            routed = m.w1.numel() + m.w2.numel()
            total -= routed - routed * m.top_k // m.n_exp
    return total
