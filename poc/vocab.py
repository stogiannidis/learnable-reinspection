"""Fixed vocabulary (~35 tokens) for the synthetic grid PoC."""

COLORS = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']  # indices 0-5
SHAPES = ['circle', 'square', 'triangle', 'star']                 # indices 0-3
RELATIONS = ['left', 'right', 'above', 'below']                   # indices 0-3

_SPECIAL = ['<PAD>', '<BOS>', '<EOS>', 'is', 'the', 'to', 'of', 'where', 'yes', 'no', 'row', 'col', 'a']
_DIGITS = [str(i) for i in range(8)]  # '0'..'7'

_ALL_TOKENS = _SPECIAL + COLORS + SHAPES + RELATIONS + _DIGITS


class Vocab:
    def __init__(self):
        self.token_to_id = {t: i for i, t in enumerate(_ALL_TOKENS)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.size = len(_ALL_TOKENS)  # 35

    @property
    def PAD(self):
        return self.token_to_id['<PAD>']

    @property
    def BOS(self):
        return self.token_to_id['<BOS>']

    @property
    def EOS(self):
        return self.token_to_id['<EOS>']

    def encode(self, tokens):
        return [self.token_to_id[t] for t in tokens]

    def decode(self, ids):
        return [self.id_to_token[i] for i in ids if i != self.PAD]


# Singleton
_vocab = None


def get_vocab():
    global _vocab
    if _vocab is None:
        _vocab = Vocab()
    return _vocab
