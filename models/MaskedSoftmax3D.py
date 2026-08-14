import torch
from torch import nn

# 这种实现直接处理多头注意力会有困难，建议使用2d展平版本
def masked_softmax_3d(x, valid_len):
    # 输入为 (batch_size, query, key)
    if valid_len is None:
        return nn.functional.softmax(x, dim=-1)

    # 输入的 valid_len有两种情况，一种为(batch_size)，即每个batch公用一个valid_len
    # 一种为(batch_size, query)
    # 需要统一转化成第二种情况
    if valid_len.dim() == 1:
        # [1,2,3] --> [1,1,2,2,3,3] --> [[1,1],[2,2],[3,3]]
        valid_len = torch.repeat_interleave(valid_len, repeats=x.shape[1]).reshape(-1, x.shape[1])

    # (1, 1, key)
    Mask3d = torch.arange(x.shape[-1], device=x.device).unsqueeze(0).unsqueeze(0)

    #(batch, query, 1)
    valid_len = valid_len.unsqueeze(-1)

    # 广播成(batch, query, key)
    MaskBool = Mask3d < valid_len

    x = x.masked_fill(~MaskBool, -torch.inf)

    return nn.functional.softmax(x, dim=-1)