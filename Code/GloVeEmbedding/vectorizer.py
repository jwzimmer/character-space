from dataclasses import dataclass

from vocabulary import Vocabulary


@dataclass
class Vectorizer:
    """
    Basically a wrapper for a Vocabulary object that has a class method to
    initialize it from a corpus and an accessor method that returns tokens
    from the corpus. Essentially a simplified interface for the underlying
    class.
    """
    vocab: Vocabulary

    @classmethod
    def from_corpus(cls, corpus: list[str], vocab_size: int = None):
        vocab = Vocabulary()
        for token in corpus:
            vocab.add(token)
        # If vocab_size is not specified as an integer, then return complete
        # Vocabulary. Otherwise, return a vocabulary consisting of the
        # vocab_size most frequently occurring tokens.
        if vocab_size is not None:
            vocab = vocab.get_topk_subset(vocab_size)
        vocab.shuffle()
        return cls(vocab)

    # takes a corpus and returns a version in which the words in Vocabulary
    # have been replaced with their indices, and words not in Vocabulary have
    # been replaced with the value used to specify words not in Vocabulary -
    # 'unknown' words.
    def vectorize(self, corpus: list[str]) -> list[int]:
        return [self.vocab[token] for token in corpus]