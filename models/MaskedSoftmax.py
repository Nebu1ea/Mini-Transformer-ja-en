import torch
from torch import nn

def masked_softmax(x, valid_len):
    if valid_len is None:
        return nn.functional.softmax(x, dim=-1)

    # 输入的 valid_len有两种情况，一种为(batch_size)，即每个batch公用一个valid_len
    # 一种为(batch_size, query)
    if valid_len.dim() == 1:
        # [1,2,3] --> [1,1,2,2,3,3]
        valid_len = torch.repeat_interleave(valid_len, repeats=x.shape[1])
    else:
        valid_len = valid_len.reshape(-1)

    OriginShape = x.shape

    x = x.reshape(-1, x.shape[-1])

    # (1, key)
    Mask = torch.arange(x.shape[-1], device=x.device).unsqueeze(0)

    # (batch_size* query, 1)
    valid_len = valid_len.unsqueeze(-1)

    # 广播成 (batch*query, key)
    MaskBool = Mask < valid_len

    x = x.masked_fill(~MaskBool, -torch.inf)

    return nn.functional.softmax(x, dim=-1).reshape(OriginShape)
