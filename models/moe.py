import torch
import torch.nn as nn
from torch.nn import functional as F
from .dense import MLP

class MoE(nn.Module):
    """ top-k mixture of experts, capacity-padded so all experts run as one bmm.
    each expert has d_ff 4C/(top_k + n_shared), so active flops match the dense 4C mlp.
    coarse (mixtral): 8 experts, top-2. fine-grained (deepseek): 64 experts, top-7 + 1 always-on shared expert """

    def __init__(self, n_embd, n_exp, top_k, n_shared=0, capacity_factor=1.25):
        super().__init__()
        d_ff = 4 * n_embd // (top_k + n_shared)
        self.n_exp = n_exp
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(n_embd, n_exp, bias=False)
        self.w1 = nn.Parameter(torch.randn(n_exp, n_embd, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_exp, d_ff, n_embd) * 0.02)
        self.shared = MLP(n_embd, n_shared * d_ff) if n_shared else None # shared experts fused into one mlp
        self.aux_loss = 0.0

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(-1, C) # (N, C)
        N = x.shape[0]
        probs = F.softmax(self.router(x), dim=-1) # (N, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1) # (N, k)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        # load balancing: fraction of routed choices * mean router prob, per expert
        frac = F.one_hot(top_i, self.n_exp).float().sum(1).mean(0) / self.top_k
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
        if self.shared is not None:
            out = out + self.shared(x)
        return out.view(B, T, C)
