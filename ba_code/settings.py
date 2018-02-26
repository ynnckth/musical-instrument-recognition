def get_segment_length(segment_seconds, sampling_rate, hop_length):
    import numpy as np
    return int(np.ceil((segment_seconds * sampling_rate)/float(hop_length)))


def get_mini_batch_length(target_seconds, segment_length, segment_seconds):
    return int(segment_length / segment_seconds * target_seconds)


# FFT settings
SAMPLING_RATE = 22050
HOP_LENGTH = 210
WINDOW_LENGTH = 2048
FMAX = SAMPLING_RATE / 2    # from some crazy math theorem

# General
SEGMENT_SECONDS = 10
MINI_BATCH_SECONDS = 2      # only integer numbers please

# General
CHANNELS = 1
MEL_DATA_POINTS = 128

# Batch settings
MINI_BATCH_SIZE = 100
NUM_MINI_BATCHES_PER_EPOCH = 5
SEGMENT_LENGTH = get_segment_length(SEGMENT_SECONDS, SAMPLING_RATE, HOP_LENGTH)  # number of data points for a segment of N seconds
MINI_SEGMENT_LENGTH = get_mini_batch_length(MINI_BATCH_SECONDS, SEGMENT_LENGTH, SEGMENT_SECONDS)  # two seconds
MINI_SEGMENTS_IN_SEGMENT = int(SEGMENT_LENGTH / MINI_SEGMENT_LENGTH)

TRAIN_TEST_RATIO = 0.8  # 0.8 of the data is used for training
VAL_TRAIN_RATIO = 0.2   # 0.2 of the whole is used for validation