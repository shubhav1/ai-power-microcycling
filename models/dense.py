import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """ multi-head causal self-attention (flash attention via sdpa) """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    """ dense gpt-2 feedforward """

    def __init__(self, n_embd, d_ff):
        super().__init__()
        self.fc = nn.Linear(n_embd, d_ff)
        self.proj = nn.Linear(d_ff, n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x), approximate='tanh'))

class Block(nn.Module):
    """ transformer block: attention then feedforward (dense or moe) """

    def __init__(self, n_embd, n_head, ffwd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = ffwd

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    """ gpt-2 small; ffwd(n_embd) builds each block's feedforward, the dense 4C mlp by default (moe.py swaps it) """

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=12, n_embd=768, ffwd=None, aux_weight=0.01):
        super().__init__()
        ffwd = ffwd or (lambda C: MLP(C, 4 * C))
        self.aux_weight = aux_weight
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, ffwd(n_embd)) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight # weight tying, as in gpt-2
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)) # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x)) # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        aux = [b.ffwd.aux_loss for b in self.blocks if hasattr(b.ffwd, 'aux_loss')]
        if aux:
            loss = loss + self.aux_weight * sum(aux) / len(aux)
        return logits, loss
