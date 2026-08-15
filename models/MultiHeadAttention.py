import torch
import DotProductAttention
from torch import nn

def split_qkv(qkv, num_heads):
    # 输入(batch_size, size, hidden), size代表q,k,v的个数
    OriginShape = qkv.shape


    # 变成(batch_size, size, heads, hidden/heads)
    qkv = qkv.reshape(OriginShape[0], OriginShape[1], num_heads, OriginShape[-1]//num_heads)

    # 变成(batch_size, heads, size, hidden/heads)
    qkv = qkv.transpose(1, 2).contiguous()

    # 变成(batch_size*heads, size, hidden/heads)
    return qkv.reshape(-1, qkv.shape[-2], qkv.shape[-1])

def merge_qkv(qkv, num_heads):
    # 输入(batch_size*heads, size, hidden/heads)
    OriginShape = qkv.shape

    # 变成(batch_size, heads, size, hidden/heads)
    qkv = qkv.reshape(OriginShape[0]//num_heads, num_heads, OriginShape[-2], OriginShape[-1])

    # 变成(batch_size, size, heads, hidden/heads)
    qkv = qkv.transpose(1, 2).contiguous()

    # 变成(batch_size, size, hidden)
    return qkv.reshape(qkv.shape[0], qkv.shape[1], -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size,key_size, value_size, hidden, d_model,  num_heads, dropout = 0.1, bias = False):
        super().__init__()
        self.num_heads = num_heads

        self.WQuery = nn.Linear(query_size, hidden, bias = bias)
        self.WKey = nn.Linear(key_size, hidden, bias = bias)
        self.WValue = nn.Linear(value_size, hidden, bias = bias)
        self.WOutput = nn.Linear(hidden, d_model, bias = bias)

        self.dropout = nn.Dropout(dropout)
        self.attention = DotProductAttention.DotProductAttention(dropout = dropout)

    def forward(self,query,key,value,valid_len = None):
        # query为(batch_size, query, query_size)
        # key为(batch_size, key, key_size)
        # value为(batch_size, value, value_size)

        query = self.WQuery(query)
        key = self.WKey(key)
        value = self.WValue(value)

        query = split_qkv(query, self.num_heads)
        key = split_qkv(key, self.num_heads)
        value = split_qkv(value, self.num_heads)

        # 处理valid_len, 在batch_size轴复制num_heads次即可
        if valid_len is not None:
            valid_len = torch.repeat_interleave(valid_len, self.num_heads, dim = 0)

        # 形状为(batch_size*num_heads, query, hidden//num_heads)
        self.MultiAttentionWeight = self.attention(query, key, value, valid_len)

        # 形状为(batch_size, query, hidden)
        AttentionWeight = merge_qkv(self.MultiAttentionWeight, self.num_heads)

        # 形状为(batch_size, query, d_model)
        return self.WOutput(AttentionWeight)

if __name__ == "__main__":
    query = torch.randn(2,3,4)
    key = torch.randn(2,3,4)
    value = torch.randn(2,3,4)
    Attention = MultiHeadAttention(4, 4, 4, 8, 4, 2)
    print(Attention)
    print(Attention.forward(query, key, value).shape)
    print(Attention.attention.AttentionWeight.shape)
