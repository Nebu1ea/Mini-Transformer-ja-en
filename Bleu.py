import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tqdm



def evaluate_bleu(encoder, decoder, dataloader, vocab, device):
    Encoder = encoder.to(device)
    Decoder = decoder.to(device)
    Encoder.eval()
    Decoder.eval()
    # ValidBlocker = tqdm.tqdm(dataloader)
    Vocab = vocab
    BleuEpoch = []

    with torch.no_grad():
        for ValidSourceTensors, ValidTargetTensors, ValidLabelTensors, ValidSourceValidLengths, ValidTargetValidLengths in dataloader:

            Bos = Vocab["<bos>"]
            Eos = Vocab["<eos>"]
            Pad = Vocab["<pad>"]

            ValidSourceTensors = ValidSourceTensors.to(device)
            ValidTargetTensors = ValidTargetTensors.to(device)
            ValidLabelTensors = ValidLabelTensors.to(device)
            ValidSourceValidLengths = ValidSourceValidLengths.to(device)
            ValidTargetValidLengths = ValidTargetValidLengths.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):

                EncoderOutput = Encoder(ValidSourceTensors, ValidSourceValidLengths)
                State = Decoder.init_state(EncoderOutput, ValidSourceValidLengths, ValidTargetValidLengths)

                DecoderInput = torch.full((ValidSourceTensors.shape[0], 1), Bos, dtype=torch.long, device=device)
                PredictionALl = []
                Finished = torch.zeros(ValidSourceTensors.shape[0], dtype=torch.bool, device=device)
                MaxLen = 100

                for i in range(MaxLen):
                    # 输出(batch_size, 1, vocab)
                    Prediction, State = Decoder(DecoderInput, State)
                    Prediction = Prediction.argmax(dim=-1)
                    PredictionALl.append(Prediction)

                    Finished |= (Prediction.squeeze(-1) == Eos)
                    DecoderInput = Prediction

                    if Finished.all():
                        break

            Sentence = torch.stack(PredictionALl, dim=1).reshape(ValidSourceTensors.shape[0], len(PredictionALl))


            for j in range(Sentence.shape[0]):
                EosIdx = torch.where(Sentence[j, :] == Eos)[0]
                if EosIdx.numel() > 0:
                    FirstEosIdx = EosIdx[0].item()
                    Sentence[j, FirstEosIdx + 1:] = Pad

            BleuBatch = []
            SentenceList = Sentence.tolist()
            LabelList = ValidLabelTensors.tolist()

            IdxToToken = Vocab.id_to_token
            Smooth = SmoothingFunction().method1



            for PredictionIdx, LabelIdx in zip(SentenceList, LabelList):

                # 转为词
                SentenceWords= []
                LabelWords = []

                for idx in PredictionIdx:
                    if idx == Eos:
                        break
                    if idx not in (Bos,Pad):
                        SentenceWords.append(IdxToToken[idx])


                for idx in LabelIdx:
                    if idx == Eos:
                        break
                    if idx not in (Bos,Pad):
                        LabelWords.append(IdxToToken[idx])


                if SentenceWords and LabelWords:
                    Score = sentence_bleu([LabelWords], SentenceWords, smoothing_function=Smooth)
                    BleuBatch.append(Score)

            BleuEpoch.append(sum(BleuBatch) / len(BleuBatch))
            # ValidBlocker.set_postfix(leu=sum(BleuBatch) / len(BleuBatch))

        BleuAve = sum(BleuEpoch) / len(BleuEpoch)
    return BleuAve
