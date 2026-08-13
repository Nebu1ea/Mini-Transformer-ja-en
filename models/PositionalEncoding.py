import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        # 公式为 Pi,2j = sin(X)  Pi,2j+1 = cos(X)    X = i/10000^(2j/d)
        # 二维空张量,后续填充位置信息
        Blocker = torch.zeros(1, max_len, d_model)

        # i表示行,创建表示行的列向量
        Molecule = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)

        Denominator = torch.pow(10000, torch.arange(0, d_model, step=2, dtype=torch.float32) / d_model)

        # (max_len,1) 除 (1, d_model/2), 自动广播, 变为 (max_len, d_model/2)形状
        X = Molecule / Denominator

        Blocker[:, :, 0::2] = torch.sin(X)
        Blocker[:, :, 1::2] = torch.cos(X)

        self.register_buffer('PositionalEncoding', Blocker)

    def forward(self, x):
        x = x + self.PositionalEncoding[:, :x.size(1), :]
        return self.dropout(x)



if __name__ == '__main__':
    po = PositionalEncoding(d_model=512, dropout=0.1)
    print(po.PositionalEncoding.shape)