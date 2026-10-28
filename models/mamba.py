import torch.nn as nn
from torch.nn import functional as F
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

class Mamba(nn.Module):
    """ mamba-2 language model from mamba-ssm (the paper's 370M config), with the same (logits, loss) interface.
    layers in attn_layer_idx become full causal attention instead (see hybrid.py) """

    def __init__(self, vocab_size, n_layer=48, n_embd=1024, attn_layer_idx=(), n_head=16):
        super().__init__()
        cfg = MambaConfig(d_model=n_embd, n_layer=n_layer, vocab_size=vocab_size, ssm_cfg={'layer': 'Mamba2'},
                          attn_layer_idx=list(attn_layer_idx), attn_cfg={'num_heads': n_head, 'causal': True}) # mha defaults to non-causal
        self.lm = MambaLMHeadModel(cfg)

    def forward(self, idx, targets):
        logits = self.lm(idx).logits # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
