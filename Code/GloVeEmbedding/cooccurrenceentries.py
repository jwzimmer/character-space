from __future__ import annotations

import os
import pickle
from collections import Counter
from dataclasses import dataclass

import h5py as h5py
import numpy as np
from tqdm import tqdm

from vectorizer import Vectorizer


@dataclass
class CoOccurrenceEntries:
    """
    Contains a vectorizer and a body of text from which to construct a
    co-occurrence matrix. Also includes a build method for constructing the
    matrix.
    The body of text consists of a list of vectorized tokens, the vectors
    constructed from the vocabulary in the vectorizer.
    """
    vectorized_corpus: list[int]
    vectorizer: Vectorizer

    # Takes a list of tokens (the corpus) and a vectorizer, and uses the
    # vectorizer to construct a list of indices corresponding to the tokens.
    # Returns a CoOccurrenceEntry instance constructed from the vectorizer
    # and vectorized corpus.
    # Note that the vectorizer needs to have an already initialized
    # vocabulary, presumably constructed from the same corpus
    @classmethod
    def setup(
            cls,
            corpus: list[str],
            vectorizer: Vectorizer
    ) -> CoOccurrenceEntries:
        return cls(
            vectorized_corpus=vectorizer.vectorize(corpus),
            vectorizer=vectorizer
        )

    # Returns True if index is NOT the value used to specify unknown tokens,
    # AND index is in the range [lower, upper]. In the event that the
    # supplied index is negative, returns true if index is not equal to the
    # value used to specify unknowns and false otherwise. Basically performs
    # data validation on indices when we're working on a specific section of
    # it.
    # This method will ignore the range if it is provided with a value for
    # lower less than 0, and simply determine whether the token is in
    # the vocabulary or not.
    def validate_index(self, index: int, lower: int, upper: int) -> bool:
        is_unk = index == self.vectorizer.vocab.unk_token
        if lower < 0:
            return not is_unk
        return not is_unk and lower <= index <= upper

    def build(
            self,
            window_size: int,
            num_partitions: int,
            chunk_size: int,
            output_directory: str = ".",
            file_name: str | None = ""
    ):
        """
        Constructs a file containing co-occurrence matrix in HDF5 binary format.
        :param window_size: The size of the window in the corpus to evaluate
        for co-occurrence. Tokens occurring this distance after and this
        distance before the token being embedded will be included.
        :param num_partitions: Number of segments to break the Vocabulary in
        self.vectorizer into.
        :param chunk_size: Size of chunks in the HDF5 dataset.
        :param output_directory: path to directory in which to store HDF5 File.
        :param file_name: name for HDF5 file, less suffix.
        """

        # Get list of indices used to mark beginning and end of sections of
        # vocabulary that will be processed in each iteration.
        partition_step = len(self.vectorizer.vocab) // num_partitions
        split_points = [0]
        while split_points[-1] + partition_step <= len(self.vectorizer.vocab):
            split_points.append(split_points[-1] + partition_step)
        split_points[-1] = len(self.vectorizer.vocab)

        file = None  # Initialized in first pass through first partition
        dataset = None  # Initialized in first pass through first partition

        # Iterate through each of the sections defined in split_points
        for partition_id in tqdm(range(len(split_points) - 1)):
            # Get indices of top and bottom of section of vocab to be processed.
            index_lower = split_points[partition_id]
            index_upper = split_points[partition_id + 1] - 1

            # Initialize Counter to store co-occurrence counts.
            # Keys will be 2-tuples of vectorized_corpus values.
            # Values will be floats that we will increase the value of as we
            # encounter co-occurrences. Value increase will be inversely
            # proportional to distance between co-occurrences.
            co_occurr_counts = Counter()

            # Iterate through tokens in vectorized_corpus and embed them if
            # the token occurs in the partition of the vocabulary currently
            # being evaluated.
            for i in tqdm(range(len(self.vectorized_corpus))):
                # We use the validation mechanism to see if the word in
                # the corpus we're looking at is in the partition of the
                # vocabulary we're currently working on.
                if not self.validate_index(
                        self.vectorized_corpus[i],
                        index_lower,
                        index_upper
                ):
                    continue

                # Get indices of window to assess for co-occurrence in the
                # corpus.
                context_lower = max(i - window_size, 0)
                context_upper = min(i + window_size + 1,
                                    len(self.vectorized_corpus))

                # Iterate through window
                for j in range(context_lower, context_upper):
                    # Do nothing if we're in the center of the window,
                    # or if context word is not in Vocabulary. (We don't care
                    # if the context word is in the partition we're working on).
                    if i == j or not self.validate_index(
                            self.vectorized_corpus[j],
                            -1,
                            -1
                    ):
                        continue
                    # Increment (or initialize) appropriate value in Counter
                    co_occurr_counts[
                        (self.vectorized_corpus[i], self.vectorized_corpus[j])
                    ] += 1 / abs(i - j)  # Decays with distance

            # Store counter data in a numpy array as long form table
            # consisting of three columns. The first column is partition word i,
            # the second is context word j, and the third is the value
            # associated with (i, j) in the co-occurrence matrix.
            co_occurr_dataset = np.zeros((len(co_occurr_counts), 3))
            for index, ((i, j), co_occurr_count) in enumerate(
                    co_occurr_counts.items()):
                co_occurr_dataset[index] = (i, j, co_occurr_count)

            # If this is our first pass through, initialize file as HDF5 file
            # that we will accumulate results in, and dataset as a dataset
            # object stored in the file.
            if partition_id == 0:
                file = h5py.File(
                    os.path.join(
                        output_directory,
                        f"{file_name}_coocurrence_dataset.hdf5"
                    ),
                    "w"
                )
                dataset = file.create_dataset(
                    "cooccurrence",
                    (len(co_occurr_counts), 3),
                    maxshape=(None, 3),
                    chunks=(chunk_size, 3)
                )
                prev_len = 0
            # Otherwise, increase the size of the dataset object so we can
            # add our new co_occur_dataset object to it.
            else:
                prev_len = dataset.len()
                # increase the number of rows of data, so that we can add the
                # data from the partition we just processed.
                dataset.resize(dataset.len() + len(co_occurr_counts), axis=0)
            # Update the appropriate rows in the dataset with our current
            # co_occurr_dataset.
            dataset[prev_len: dataset.len()] = co_occurr_dataset

        file.close()

        # Store vocabulary as a pickled file.
        with open(
                os.path.join(output_directory, f"{file_name}_vocab.pickle"),
                "wb"
        ) as file:
            pickle.dump(self.vectorizer.vocab, file)
