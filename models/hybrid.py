from .mamba import Mamba

class Hybrid(Mamba):
    """ mamba-2 with full attention at every 8th layer: 3 of 24 (the mamba-2 paper found ~10% attention works best).
    no positional embeddings, the mamba layers carry position (as in jamba) """

    def __init__(self, vocab_size, n_layer=24, n_embd=768, n_head=12, attn_every=8):
        super().__init__(vocab_size, n_layer, n_embd, range(attn_every - 1, n_layer, attn_every), n_head)
