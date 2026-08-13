import collections


def count_tokens(tokens):
    if not tokens:
        return collections.Counter()


    # [["1"],["1","2","3"]]] --> ["1","1","2","3"]
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]


    # 返回 ("1": 2, "2": 1, "3": 1)
    return collections.Counter(tokens)


class Vocabulary:
    def __init__(self, tokens, min_freq, reserved_tokens = None):
        if not tokens:
            tokens = []

        if not reserved_tokens:
            reserved_tokens = []

        token_count = count_tokens(tokens)

        # sort the count
        self._token_freq = sorted(token_count.items(), key = lambda x: x[1], reverse = True)

        self.id_to_token = ["<unk>"] + reserved_tokens
        self.token_to_id = {
            token: idx for idx, token in enumerate(self.id_to_token)
        }

        # 记录出现频率大于最小频率的token
        for token, freq in self._token_freq:
            if freq < min_freq:
                pass
            else:
                self.id_to_token.append(token)
                self.token_to_id[token] = len(self.id_to_token) - 1


    def __len__(self):
        return len(self.id_to_token)

    # 获取token编号
    def __getitem__(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.__getitem__(token) for token in tokens]
        return self.token_to_id.get(tokens, self.unk)

    def token(self, idx):
        return self.id_to_token[idx]

    # 只读属性unk编号
    @property
    def unk(self):
        return 0

    #只读属性token_freq
    @property
    def token_freq(self):
        return self._token_freq