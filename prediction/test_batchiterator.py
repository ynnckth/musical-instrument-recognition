import numpy as np
from nolearn.lasagne import BatchIterator

import utils.utils as utils


class TestSegmentBatchIterator(BatchIterator):
    def __init__(self, batch_size):
        super(TestSegmentBatchIterator, self).__init__(batch_size)

    def __iter__(self):
        mini_batch_size = self.batch_size
        if self.y is None:
            # y doesn't matter at all. can be random -> is not used for predicting
            self.y = np.zeros(len(self.X), dtype=np.int32)
            Xb = self.X
            yb = self.y
        else:
            Xb, yb = utils.extract_mini_segments(self.X, self.y)
            Xb, yb = utils.filter_mini_segments(Xb, yb)
        # loop through all test data
        for i in range((len(Xb) + mini_batch_size - 1) // mini_batch_size):
            mini_batch_indices = slice(i * mini_batch_size, (i + 1) * mini_batch_size)

            yield self.transform(Xb[mini_batch_indices], yb[mini_batch_indices])
