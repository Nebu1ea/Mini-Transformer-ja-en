import torch
from torch import nn
from . import MaskedSoftmax
import math

class DotProductAttention(nn.Module):
    def __init__(self, dropout = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_len = None):
        # 公式为 softmax(x)*value
        # x = (query * key^T) / sqrt(d_model)

        # query为(batch_size, query, d_model)
        # key为(batch_size, key, d_model)
        # value为(batch_size, value, d_model)


        # Score为(batch, query, key)
        Score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        self.AttentionWeight = MaskedSoftmax.masked_softmax(Score, valid_len)

        return torch.bmm(self.dropout(self.AttentionWeight), value)


if __name__ == "__main__":
    x = torch.randn(2, 3, 4)
    attention = DotProductAttention()

    print(attention.forward(x, x, x).shape)