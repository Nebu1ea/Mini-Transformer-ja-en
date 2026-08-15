from torch import nn
import AddNorm
import MultiHeadAttention
import PositionWiseFeedForwardNet


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.MultiHeadAttention = MultiHeadAttention.MultiHeadAttention(query_size=d_model, key_size=d_model, value_size=d_model, hidden=d_model, d_model=d_model, num_heads=num_heads, dropout = dropout)
        self.AddNorm1 = AddNorm.AddNorm(d_model=d_model, dropout=dropout)
        self.AddNorm2 = AddNorm.AddNorm(d_model=d_model, dropout=dropout)
        self.Ffn = PositionWiseFeedForwardNet.PositionWiseFFN(d_model=d_model, d_ff= d_ff, dropout=dropout)


    def forward(self, x, valid_len = None):
        # 输入(batch_size, steps, d_model)
        OriginX = x

        # 自注意力
        x = self.MultiHeadAttention(x, x, x, valid_len=valid_len)
        x = self.AddNorm1(x, OriginX)
        OriginX = x

        x = self.Ffn(x)
        x = self.AddNorm2(x, OriginX)

        return x