import torch
from torch import nn
import MultiHeadAttention
import AddNorm
import PositionWiseFeedForwardNet

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, i = 0, bias = False, dropout=0.1):
        super().__init__()

        self.MaskedAttention = MultiHeadAttention.MultiHeadAttention(query_size=d_model, key_size=d_model, value_size=d_model, hidden=d_model, d_model=d_model, num_heads=num_heads, dropout = dropout, bias=bias)
        self.AddNorm1 = AddNorm.AddNorm(d_model, dropout)

        self.Attention = MultiHeadAttention.MultiHeadAttention(query_size=d_model, key_size=d_model, value_size=d_model, hidden=d_model, num_heads=num_heads, d_model=d_model, dropout = dropout, bias=bias)
        self.AddNorm2 = AddNorm.AddNorm(d_model, dropout)

        self.Ffn = PositionWiseFeedForwardNet.PositionWiseFFN(d_ff = d_ff,d_model=d_model,  dropout = dropout)
        self.AddNorm3 = AddNorm.AddNorm(d_model, dropout)

        self.i = i

    def forward(self, x, state):

        EncoderOutput, EncoderValidLen, KvCache, TargetValidLen = state

        # KvCache，保存已生成的K和V,推理的时候才有用
        # 训练阶段，KvCache[self.i] is None恒成立
        if KvCache[self.i] is None:
            KvCache[self.i] = x
        else:
            KvCache[self.i] = torch.cat((KvCache[self.i], x), dim = 1)

        state[2][self.i] = KvCache[self.i]

        if self.training:
            # 原句: I am Nebu1ea <pad> <pad>
            # I -> I
            # am -> I am
            # Nebu1ea -> I am Nebu1ea
            # <pad> -> I am Nebu1ea
            # <pad> -> I am Nebu1ea
            DecoderValidLen = torch.arange(1, x.shape[1] + 1, device=x.device).repeat(x.shape[0]).reshape(x.shape[0], x.shape[1])
            DecoderValidLen = torch.minimum(DecoderValidLen, TargetValidLen)
        else:
            DecoderValidLen = None

        OriginX = x
        x = self.MaskedAttention(x, KvCache[self.i], KvCache[self.i], DecoderValidLen)
        x = self.AddNorm1(x, OriginX)

        OriginX = x
        # 交叉注意力
        x = self.Attention(x, EncoderOutput, EncoderOutput, EncoderValidLen)
        x = self.AddNorm2(x, OriginX)

        OriginX = x
        x = self.Ffn(x)
        x = self.AddNorm3(x, OriginX)

        return x, state

