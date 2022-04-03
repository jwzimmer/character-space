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
    vectorized_corpus: list
    vectorizer: Vectorizer

    @classmethod
    def setup(cls, corpus: list[str], vectorizer: Vectorizer):
        return cls(
            vectorized_corpus=vectorizer.vectorize(corpus),
            vectorizer=vectorizer
        )

    # Returns True if index is not the value used to specify unknown tokens,
    # and index is in the range [lower, upper]. In the event that the
    # supplied index is negative, returns true if index is not equal to the
    # value used to specify unknowns and false otherwise.
    def validate_index(self, index: int, lower: int, upper: int):
        is_unk = index == self.vectorizer.vocab.unk_token
        if lower < 0:
            return not is_unk
        return not is_unk and lower <= index <= upper

    # Construct file containing co-occurrence matrix in HDF5 binary format
    def build(
            self,
            window_size: int,
            num_partitions: int,
            chunk_size: int,
            output_directory: str = "."
    ):
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
            # Keys will be tuples of vectorized_corpus values.
            # Values will be floats that we will increase the value of as we
            # encounter co-occurrences. Value increase will be inversely
            # proportional to distance between co-occurrences.
            co_occurr_counts = Counter()

            # Iterate through indices of vectorized_corpus, within the
            # boundaries specified by index_lower and index_upper.
            for i in tqdm(range(len(self.vectorized_corpus))):
                # Kind of an odd way to limit range...
                if not self.validate_index(
                        self.vectorized_corpus[i],
                        index_lower,
                        index_upper
                ):
                    continue

                # Get indices of window to assess for co-occurrence
                # Note that this allows for inclusion of values outside the
                # current partition being processed.
                context_lower = max(i - window_size, 0)
                context_upper = min(i + window_size + 1,
                                    len(self.vectorized_corpus))

                # Iterate through window
                for j in range(context_lower, context_upper):
                    # Do nothing if we're in the center of the window,
                    # or if word is not in Vocabulary.
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
            # consisting of three columns. The first column is word i,
            # the second is word j, and the third is the value connecting j
            # to i in the co-occurrence matrix.
            co_occurr_dataset = np.zeros((len(co_occurr_counts), 3))
            for index, ((i, j), co_occurr_count) in enumerate(
                    co_occurr_counts.items()):
                co_occurr_dataset[index] = (i, j, co_occurr_count)

            # If this is our first pass through, initialize file as HDF5 file
            # that we will accumulate results in.
            if partition_id == 0:
                file = h5py.File(
                    os.path.join(
                        output_directory,
                        "cooccurrence.hdf5"
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
            # Otherwise, update our HDF5 file with the new data.
            else:
                prev_len = dataset.len()
                dataset.resize(dataset.len() + len(co_occurr_counts), axis=0)
            dataset[prev_len: dataset.len()] = co_occurr_dataset

        file.close()

        # Store HDF5 file and vocabulary as single pickled file.
        with open(os.path.join(output_directory, "vocab.pkl"), "wb") as file:
            pickle.dump(self.vectorizer.vocab, file)