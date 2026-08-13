from torch.utils.data import Dataset
import torch

# 截断长度太长的，填充长度不够的
def truncate(sentence, num_steps, padding):
    if len(sentence) > num_steps:
        return sentence[:num_steps]
    return sentence + [padding] * (num_steps - len(sentence))


class TextDataset(Dataset):
    def __init__(self,source, target, num_step, source_vocab, target_vocab):
        super().__init__()
        self.SourceTokensId = source
        self.TargetTokensId = target
        self.NumStep = num_step
        self.Padding = source_vocab["<pad>"]
        self.Bos = source_vocab["<bos>"]
        self.Eos = source_vocab["<eos>"]

        self.SourceTensors = []
        self.TargetTensors = []
        self.LabelTensors = []
        self.SourceValidLengths = []
        self.TargetValidLengths = []


        # encoder的输入，把原始张量变成 --> [<bos>, 序列, <eos>, <pad>]
        for SourceTokenId in self.SourceTokensId:
            FullSourceTokenId = [self.Bos] + SourceTokenId

            if len(FullSourceTokenId) > self.NumStep - 1:
                FullSourceTokenId = truncate(FullSourceTokenId, self.NumStep - 1, self.Padding)
                FullSourceTokenId.append(self.Eos)
                self.SourceValidLengths.append(len(FullSourceTokenId))

            else:
                FullSourceTokenId.append(self.Eos)
                self.SourceValidLengths.append(len(FullSourceTokenId))
                FullSourceTokenId = truncate(FullSourceTokenId, self.NumStep, self.Padding)

            self.SourceTensors.append(torch.tensor(FullSourceTokenId, dtype=torch.long))

        # decoder的输入，transformer自回归生成下一个token，所以删除最后的<eos>
        # [<bos>, 序列, <pad>]
        for TargetTokenId in self.TargetTokensId:
            FullTargetTokenId = [self.Bos] + TargetTokenId
            OriginLength = len(FullTargetTokenId)
            FullTargetTokenId = truncate(FullTargetTokenId, self.NumStep, self.Padding)

            if OriginLength > self.NumStep:
                self.TargetValidLengths.append(self.NumStep)
            else:
                self.TargetValidLengths.append(OriginLength)

            self.TargetTensors.append(torch.tensor(FullTargetTokenId, dtype=torch.long))


        # 计算损失的张量，根据自回归,模型输出为[序列,<eos>,<pad>(这里的<pad>指要掩盖的东西，而非模型真的输出<pad>)]
        # [序列, <eos>, <pad>]

        for LabelTokenId in self.TargetTokensId:
            FullLabelTokenId = list(LabelTokenId)
            if len(FullLabelTokenId) > self.NumStep - 1:
                FullLabelTokenId = truncate(FullLabelTokenId, self.NumStep - 1, self.Padding)
                FullLabelTokenId.append(self.Eos)
            else:
                FullLabelTokenId.append(self.Eos)
                FullLabelTokenId = truncate(FullLabelTokenId, self.NumStep, self.Padding)

            self.LabelTensors.append(torch.tensor(FullLabelTokenId, dtype=torch.long))

    def __len__(self):
        return len(self.SourceTensors)

    def __getitem__(self, idx):
        return self.SourceTensors[idx], self.TargetTensors[idx], self.LabelTensors[idx], self.SourceValidLengths[idx], self.TargetValidLengths[idx]
