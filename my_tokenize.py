import nltk
import pandas as pd
from janome.tokenizer import Tokenizer
from nltk.tokenize import word_tokenize
from concurrent.futures import ProcessPoolExecutor
import time
import pickle

JaTokenize = None

def jaInitWork():
    global JaTokenize
    JaTokenize = Tokenizer(wakati=True)


def jaTokenize(text):
    return list(JaTokenize.tokenize(text))


def enTokenize(text):
    return word_tokenize(text)



def main():
    nltk.download('punkt_tab')


    CleanedDataDir = "./data/cleaned.txt"
    SecondPartData = "./data/日语-英语.tsv"
    JaTokensFile = "./data/JaTokens.pkl"
    EnTokensFile = "./data/EnTokens.pkl"

    Source = []
    Target = []

    # 读上方处理过的txt文件
    with open(CleanedDataDir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            parts = line.split('\t')
            Source.append(parts[0])
            Target.append(parts[1])

    # 处理第二部分tsv文件, 分隔符为'\t'，把引号当成普通字符串处理
    pd_tsv = pd.read_csv(SecondPartData, sep="\t", quoting=3, header=None).fillna("")
    SecondPartJA = pd_tsv.iloc[:, 1].tolist()
    SecondPartEN = pd_tsv.iloc[:, 3].tolist()
    Source = Source + SecondPartJA
    Target = Target + SecondPartEN

    print(len(Source))
    print(len(Target))

    JaTokens = []
    EnTokens = []

    JaStartTime = time.time()

    with ProcessPoolExecutor(initializer=jaInitWork, max_workers=8) as executor:
        JaTokens = list(executor.map(jaTokenize, Source, chunksize=2000))

    JaEndTime = time.time()

    print("successed in "+ str(JaStartTime - JaEndTime))

    EnStartTime = time.time()

    with ProcessPoolExecutor(max_workers=8) as executor:
        EnTokens = list(executor.map(enTokenize, Target, chunksize=2000))

    EnEndTime = time.time()

    print("successed in "+ str(EnStartTime - EnEndTime))

    with open(JaTokensFile, "wb") as f:
        pickle.dump(JaTokens, f)

    with open(EnTokensFile, "wb") as f:
        pickle.dump(EnTokens, f)

    print(JaTokens[:50])
    print(EnTokens[:50])


if __name__ == "__main__":
    main()