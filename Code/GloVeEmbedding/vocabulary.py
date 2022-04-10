from __future__ import annotations

from dataclasses import field, dataclass

import numpy as np


@dataclass
class Vocabulary:
    """
    Stores all the unique tokens occurring in a corpus, along with counts of
    the number of times they occur. Includes useful accessors and mutators,
    and methods for shuffling order of data and obtaining subsets of the most
    frequently occurring tokens of arbitrary size as new Vocabulary objects.
    """
    # To get index from token
    token2index: dict = field(default_factory=dict)
    # To get token from index
    index2token: dict = field(default_factory=dict)
    # todo: convert token_counts to dictionary
    # Number of occurrence of tokens by index (should be a dictionary)
    token_counts: list = field(default_factory=list)
    # Value used to specify unknown tokens
    _unk_token: int = field(init=False, default=-1)

    # Tell Vocabulary object it has encountered a new token.
    # If novel, add to lookup dictionaries and token_counts.
    # If not novel, just increment appropriate index of token_counts.
    def add(self, token: str) -> None:
        if token not in self.token2index:
            index = len(self)
            self.token2index[token] = index
            self.index2token[index] = token
            self.token_counts.append(0)
        self.token_counts[self.token2index[token]] += 1

    # Returns a new Vocabulary object containing only the k most frequently
    # occurring tokens, along with their counts in the original Vocabulary.
    def get_topk_subset(self, k: int) -> Vocabulary:
        tokens = sorted(
            list(self.token2index.keys()),
            key=lambda token: self.token_counts[self[token]],
            reverse=True
        )
        return Vocabulary(
            token2index={token: index for index, token in
                         enumerate(tokens[:k])},
            index2token={index: token for index, token in
                         enumerate(tokens[:k])},
            token_counts=[
                self.token_counts[self.token2index[token]] for token in
                tokens[:k]
            ]
        )

    # Randomizes the order in which tokens are stored. Useful because in
    # construction of Vocabulary from corpus tends to result in accumulation
    # of high frequency tokens at beginning of Vocabulary. Shuffling
    # generates uniform density - useful for co-occurrence matrix
    # construction, where matrix can become extremely large and may need to
    # be processed in segments.
    def shuffle(self) -> None:
        new_index = [i for i in range(len(self))]
        np.random.shuffle(new_index)
        new_token_counts = [None] * len(self)
        for token, index in zip(list(self.token2index.keys()), new_index):
            new_token_counts[index] = self.token_counts[self[token]]
            self.token2index[token] = index
            self.index2token[index] = token
        self.token_counts = new_token_counts

    # Returns index of token.
    def get_index(self, token: str) -> int:
        return self[token]

    # Returns token at index
    def get_token(self, index: int) -> str:
        if not index in self.index2token:
            raise IndexError("Invalid index.")
        return self.index2token[index]

    # Returns the value being used to specify unknown tokens
    @property
    def unk_token(self) -> int:
        return self._unk_token

    # Returns index of token IF token is in Vocabulary, else returns value
    # associated with unknown token.
    def __getitem__(self, token: str) -> int:
        if token not in self.token2index:
            return self._unk_token
        return self.token2index[token]

    # Returns number of unique tokens in Vocabulary.
    def __len__(self) -> int:
        return len(self.token2index)
