import torch
from torch import nn

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout = 0.1,):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    # 公式为 Y = f(x) + X
    def forward(self, x, origin_x):
        origin_x = self.dropout(origin_x)
        x = x + origin_x
        x = self.norm(x)
        return x

if __name__ == '__main__':
    x = torch.randn(4, 5)
    origin_x = torch.randn(4, 5)
    AddNorm = AddNorm(5)
    print(AddNorm.forward(x, origin_x).shape)