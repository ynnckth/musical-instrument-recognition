import numpy as np

import ba_code.utils.utils as utils
from nolearn.lasagne import BatchIterator
from ba_code import settings
from ba_code.utils.spectrogram_tools import spectrogram_utils

# threshold = max number of zeros tolerated
MINI_SEGMENT_ZERO_THRESHOLD = int(settings.MINI_SEGMENT_LENGTH * settings.MEL_DATA_POINTS * 0.25)
MINI_SEGMENT_SHIFT_STEP = settings.MINI_SEGMENT_LENGTH / 2


class TrainSegmentBatchIterator(BatchIterator):
    def __init__(self, batch_size):
        super(TrainSegmentBatchIterator, self).__init__(batch_size)

    def __iter__(self):
        mini_batch_size = self.batch_size    # our batch size has 100 segments

        # loop through training data and generate mini-batches
        for mini_batch in range(settings.NUM_MINI_BATCHES_PER_EPOCH):  # our pseudo-epoch has 5 mini-batches

            X_mini_batch = np.zeros((mini_batch_size, 1, settings.MINI_SEGMENT_LENGTH, settings.MEL_DATA_POINTS), dtype=np.float32)
            y_mini_batch = np.zeros(mini_batch_size, dtype=np.int32)

            # here one mini-batch is filled
            for i in range(mini_batch_size):

                # select a random segment from the training data
                random_segment_idx = self.random.randint(0, len(self.y))

                # extract a random mini segment from this segment
                spectrogram_utils.save_spectrogram_plot(np.transpose(self.X[random_segment_idx, 0]), "/home/ba16-stdm-streit/temp/spect_10s.png")

                X_mini_batch[i] = self.extract_random_mini_segment(self.X[random_segment_idx])
                y_mini_batch[i] = self.y[random_segment_idx]

            yield self.transform(X_mini_batch, y_mini_batch)

    def extract_random_mini_segment(self, segment):
        start_idx = self.random.randint(0, settings.SEGMENT_LENGTH - settings.MINI_SEGMENT_LENGTH)
        end_idx = start_idx + settings.MINI_SEGMENT_LENGTH
        mini_segment = segment[0, start_idx:end_idx]

        zeros_in_mini_segment = utils.get_zeros_in_segment(mini_segment)

        # shift to left and extract mini segment
        while zeros_in_mini_segment > MINI_SEGMENT_ZERO_THRESHOLD:
            if start_idx <= 0:
                start_idx = 0
                end_idx = start_idx + settings.MINI_SEGMENT_LENGTH
                mini_segment = segment[0, start_idx:end_idx]
                break
            start_idx -= MINI_SEGMENT_SHIFT_STEP
            end_idx = start_idx + settings.MINI_SEGMENT_LENGTH
            mini_segment = segment[0, start_idx:end_idx]
            zeros_in_mini_segment = utils.get_zeros_in_segment(mini_segment)

        spectrogram_utils.save_spectrogram_plot(np.transpose(mini_segment), "/home/ba16-stdm-streit/temp/spect_2s.png")
        return mini_segment

