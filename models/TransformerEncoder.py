import math

from torch import nn
from . import PositionalEncoding
from . import TransformerEncoderBlock


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, layers, dropout = 0.1, bias = False):
        super().__init__()
        self.AttentionWeights = [None] * layers
        self.d_model = d_model
        self.Embedding = nn.Embedding(vocab_size, d_model)
        self.PositionalEncoding = PositionalEncoding.PositionalEncoding(d_model, dropout)
        self.Blockers = nn.ModuleList([TransformerEncoderBlock.TransformerEncoderBlock(d_model, num_heads, d_ff, dropout, bias) for _ in range(layers)])

    def forward(self, x, valid_len = None):
        x = self.Embedding(x) * math.sqrt(self.d_model)
        x = self.PositionalEncoding(x)

        for i , block in enumerate(self.Blockers):
            x = block(x, valid_len)
            self.AttentionWeights[i] = block.MultiHeadAttention.MultiAttentionWeight

        return x

if __name__ == "__main__":
    d_model = 512
    num_heads = 8
    d_ff = 2048
    dropout = 0.2
    bias = True

    transformer = TransformerEncoder(1, d_model, num_heads, d_ff, 2, dropout, bias)