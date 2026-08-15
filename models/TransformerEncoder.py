import math

from torch import nn
import PositionalEncoding
import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ffn, layers, dropout = 0.1, bias = False):
        super().__init__()
        self.AttentionWeights = [None] * layers
        self.d_model = d_model
        self.Embedding = nn.Embedding(vocab_size, d_model)
        self.PositionalEncoding = PositionalEncoding.PositionalEncoding(d_model, dropout)
        self.Blockers = nn.ModuleList([TransformerEncoderBlock.TransformerEncoderBlock(d_model, num_heads, d_ffn, dropout) for _ in range(layers)])

    def forward(self, x, valid_len = None):
        x = self.Embedding(x) * math.sqrt(self.d_model)
        x = self.PositionalEncoding(x)

        for i , block in enumerate(self.Blockers):
            x = block(x, valid_len)
            self.AttentionWeights[i] = block.MultiHeadAttention.MultiAttentionWeight

        return x
