from torch import nn

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.Linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.Linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.Linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.Linear2(x)
        return x

