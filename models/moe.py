import torch
import torch.nn as nn
from torch.nn import functional as F
from .dense import MLP

class MoE(nn.Module):
    """ dropless top-k mixture of experts: tokens are sorted by expert and all experts run in one grouped gemm
    (as in torchtitan / megatron), so no capacity padding and no dropped tokens.
    each expert has d_ff 4C/(top_k + n_shared), so active flops match the dense 4C mlp.
    coarse (mixtral): 8 experts, top-2. fine-grained (deepseek): 64 experts, top-7 + 1 always-on shared expert """

    align = 16 # grouped gemm wants each expert's group of rows to start on a multiple of 16

    def __init__(self, n_embd, n_exp, top_k, n_shared=0):
        super().__init__()
        d_ff = 4 * n_embd // (top_k + n_shared)
        self.n_exp = n_exp
        self.top_k = top_k
        self.router = nn.Linear(n_embd, n_exp, bias=False)
        self.w1 = nn.Parameter(torch.randn(n_exp, n_embd, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_exp, d_ff, n_embd) * 0.02)
        self.shared = MLP(n_embd, n_shared * d_ff) if n_shared else None # shared experts fused into one mlp
        self.aux_loss = 0.0

    def expert_mm(self, x, w, ends):
        """ rows of x grouped by expert, group e ending at row ends[e]. one grouped gemm on the gpu, a loop on cpu (for testing) """
        if x.is_cuda:
            return torch._grouped_mm(x, w.to(x.dtype), offs=ends.int())
        out, start = [], 0
        for e, end in enumerate(ends.tolist()):
            out.append(x[start:end] @ w[e])
            start = end
        return torch.cat(out)

    def forward(self, x):
        B, T, C = x.shape
        x = x.view(-1, C) # (N, C)
        N = x.shape[0]
        probs = F.softmax(self.router(x), dim=-1) # (N, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1) # (N, k)
        top_p = top_p / top_p.sum(dim=-1, keepdim=True)

        # sort every (token, choice) by expert. nothing here reads a gpu value on the host, so no syncs mid-step
        exp_i = top_i.view(-1) # (N*k,)
        tok_i = torch.arange(N, device=x.device).repeat_interleave(self.top_k) # (N*k,)
        gate = top_p.view(-1) # (N*k,)
        order = exp_i.argsort(stable=True)
        exp_i, tok_i, gate = exp_i[order], tok_i[order], gate[order]
        counts = torch.zeros(self.n_exp, dtype=torch.long, device=x.device).index_add_(0, exp_i, torch.ones_like(exp_i)) # (E,)

        # load balancing: fraction of routed choices * mean router prob, per expert
        self.aux_loss = self.n_exp * (counts / (N * self.top_k) * probs.mean(0)).sum()

        # row of each (token, choice) in a buffer where expert e's group is padded up to a multiple of align.
        # the buffer has a static upper-bound size; the last expert's group absorbs the unused tail (all zeros)
        padded = (counts + self.align - 1) // self.align * self.align
        ends = padded.cumsum(0)
        rank = torch.arange(len(exp_i), device=x.device) - (counts.cumsum(0) - counts)[exp_i] # position within its expert
        dest = (ends - padded)[exp_i] + rank
        M = N * self.top_k + self.n_exp * self.align
        ends[-1] = M

        dtype = torch.bfloat16 if x.is_cuda else x.dtype # grouped gemm is bf16 only
        buf = x.new_zeros(M, C, dtype=dtype)
        buf[dest] = x[tok_i].to(dtype)
        h = F.gelu(self.expert_mm(buf, self.w1, ends), approximate='tanh') # (M, d_ff)
        y = self.expert_mm(h, self.w2, ends) # (M, C)

        out = torch.zeros_like(x)
        out.index_add_(0, tok_i, (y[dest] * gate[:, None]).to(x.dtype))
        if self.shared is not None:
            out = out + self.shared(x)
        return out.view(B, T, C)
