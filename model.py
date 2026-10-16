import math
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

class MoE(nn.Module):
    """ top-k mixture of experts, capacity-padded so all experts run as one bmm """

    def __init__(self, n_embd, d_ff, n_exp, top_k, capacity_factor):
        super().__init__()
        self.n_exp = n_exp
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(n_embd, n_exp, bias=False)
        self.w1 = nn.Parameter(torch.randn(n_exp, n_embd, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_exp, d_ff, n_embd) * 0.02)
        self.aux_loss = 0.0

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(-1, C) # (N, C)
        N = x.shape[0]
        probs = F.softmax(self.router(x), dim=-1) # (N, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1) # (N, k)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        # switch-style load balancing: fraction of first choices * mean router prob, per expert
        frac = F.one_hot(top_i[:, 0], self.n_exp).float().mean(0)
        self.aux_loss = self.n_exp * (frac * probs.mean(0)).sum()

        # slot of each (choice, token) in its expert's queue; first choices fill queues first
        cap = int(self.capacity_factor * N * self.top_k / self.n_exp)
        exp_i = top_i.t().reshape(-1) # (k*N,)
        tok_i = torch.arange(N, device=x.device).repeat(self.top_k) # (k*N,)
        gate = top_p.t().reshape(-1) # (k*N,)
        onehot = F.one_hot(exp_i, self.n_exp) # (k*N, E)
        slot = ((onehot.cumsum(0) - 1) * onehot).sum(-1) # (k*N,)
        keep = slot < cap # overflow tokens are dropped
        exp_i, tok_i, gate, slot = exp_i[keep], tok_i[keep], gate[keep], slot[keep]

        expert_in = x.new_zeros(self.n_exp, cap, C)
        expert_in[exp_i, slot] = x[tok_i]
        h = F.gelu(torch.bmm(expert_in, self.w1), approximate='tanh') # (E, cap, d_ff)
        expert_out = torch.bmm(h, self.w2) # (E, cap, C)

        out = torch.zeros_like(x)
        out.index_add_(0, tok_i, expert_out[exp_i, slot] * gate[:, None].to(x.dtype))
        return out.view(B, T, C)

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
    """ gpt-2 small; moe=True swaps every feedforward for an active-flop-matched moe """

    def __init__(self, vocab_size, block_size, n_layer=12, n_head=12, n_embd=768, moe=False,
                 n_exp=8, top_k=2, capacity_factor=1.25, aux_weight=0.01):
        super().__init__()
        self.aux_weight = aux_weight
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        blocks = []
        for _ in range(n_layer):
            if moe:
                # top_k experts of d_ff 4C/top_k = same active flops as the dense 4C mlp
                ffwd = MoE(n_embd, 4 * n_embd // top_k, n_exp, top_k, capacity_factor)
            else:
                ffwd = MLP(n_embd, 4 * n_embd)
            blocks.append(Block(n_embd, n_head, ffwd))
        self.blocks = nn.ModuleList(blocks)
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
        aux = [b.ffwd.aux_loss for b in self.blocks if isinstance(b.ffwd, MoE)]
        if aux:
            loss = loss + self.aux_weight * sum(aux) / len(aux)
        return logits, loss

class Mamba(nn.Module):
    """ mamba-2 language model from mamba-ssm, with the same (logits, loss) interface """

    def __init__(self, vocab_size, n_layer=24, n_embd=768):
        super().__init__()
        # imported here so dense/moe runs don't need mamba-ssm installed
        from mamba_ssm.models.config_mamba import MambaConfig
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
        cfg = MambaConfig(d_model=n_embd, n_layer=n_layer, vocab_size=vocab_size, ssm_cfg={'layer': 'Mamba2'})
        self.lm = MambaLMHeadModel(cfg)

    def forward(self, idx, targets):
        logits = self.lm(idx).logits # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def active_params(model):
    """ params touched per token (moe counts top_k of n_exp experts) """
    total = 0
    for name, p in model.named_parameters():
        if name.endswith('w1') or name.endswith('w2'):
            moe = model.blocks[0].ffwd
            total += p.numel() * moe.top_k // moe.n_exp
        else:
            total += p.numel()
    return total
