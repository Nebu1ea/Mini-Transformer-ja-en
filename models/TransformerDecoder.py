import math

from torch import nn
import PositionalEncoding
import TransformerDecoderBlock


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, layers,dropout = 0.1, bias = False):
        super().__init__()
        self.d_model = d_model

        self.Embedding = nn.Embedding(vocab_size, d_model)
        self.PositionalEncoding = PositionalEncoding.PositionalEncoding(d_model)

        self.Blockers = nn.ModuleList(TransformerDecoderBlock.TransformerDecoderBlock(d_model, num_heads, d_ff, i = i, dropout=dropout, bias=bias) for i in range(layers))

        self.Linear = nn.Linear(d_model, vocab_size)

        self.MaskedAttentionWeights = [None for _ in range(layers)]
        self.AttentionWeights = [None for _ in range(layers)]
    def forward(self, x, states):

        x = self.Embedding(x)* math.sqrt(self.d_model)
        x = self.PositionalEncoding(x)

        for i, block in enumerate(self.Blockers):
            x, states = block(x, states)
            self.MaskedAttentionWeights[i] = block.MaskedAttention.MultiAttentionWeight
            self.AttentionWeights[i] = block.Attention.MultiAttentionWeight

        x = self.Linear(x)
        return x, states

    def init_state(self, encoder_outputs, encoder_valid_lens, target_valid_lens=None):
        KvCache = [None for _ in range(len(self.Blockers))]
        return [encoder_outputs, encoder_valid_lens, KvCache, target_valid_lens]