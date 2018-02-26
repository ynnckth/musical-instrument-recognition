# This module is used to load the raw audio files from the file database
# and extract the input data for the neural network. The data is divided
# into two sets: the training- and the test set. The validation set can
# later be extracted from the training set.
# The extracted sets are then saved as in a lightweight format (e.g. pickle)
# which is loaded by the cnn.

import os
import sys
import random
import numpy as np

from ba_code.utils import utils
from ba_code import settings
from ba_code.utils.spectrogram_tools import spectrogram_utils
from ba_code.utils.spectrogram_tools.spectrogram_utils import segment_signal

DATA_ROOT_DIR = "/home/ba16-stdm-streit/data/"
DATASET_ROOT_DIR = "/home/ba16-stdm-streit/datasets/"


def generate_datasets(instrument_files):
    print "Generating datasets..."
    sys.stdout.flush()
    instrument_segments = dict()

    for instrument_class, wav_file_paths in instrument_files.items():
        print "start extracting segments for {0}".format(instrument_class)
        segments = get_spectrograms_from_filepaths(wav_file_paths, instrument_class)
        print "finished extracting {0}".format(instrument_class)
        instrument_segments[instrument_class] = segments

    print
    instrument_segments = shuffle_segments(instrument_segments)
    instrument_segments = filter_zero_segments(instrument_segments)
    return split_sets(instrument_segments)


def get_spectrograms_from_filepaths(wav_file_paths, instrument_class):
    instrument_segments = []
    file_counter = 0
    list_idx = 0
    for wav_file_path in wav_file_paths:
        signal, sampling_rate = get_signal_from_file(wav_file_path)
        # split wav_file into 10 second segments
        segments = segment_signal(signal, sampling_rate, settings.SEGMENT_SECONDS)
        for segment_idx in range(len(segments)):
            # for each segment generate spectrogram
            spectrogram = extract_spectrogram(segments[segment_idx], sampling_rate)
            spectrogram = dynamic_compression(spectrogram)
            spectrogram = np.transpose(spectrogram)
            # remove last frame. see librosa:util/utils.py:91
            spectrogram = np.delete(spectrogram, [len(spectrogram) - 1], 0)
            if not np.count_nonzero(spectrogram):
                print "zero spectrogram in " + str(wav_file_path)
            instrument_segments.append([])
            instrument_segments[list_idx].append(spectrogram)
            instrument_segments[list_idx].append(wav_file_path)
            instrument_segments[list_idx].append(segment_idx)
            list_idx += 1
        file_counter += 1
        if file_counter % 10 == 0:
            print "{0} got [{1} / {2}]".format(instrument_class, file_counter, len(wav_file_paths))
            sys.stdout.flush()
    return instrument_segments


def dynamic_compression(spectrogram):
    for i in range(len(spectrogram)):
        for j in range(len(spectrogram[i])):
            spectrogram[i][j] = np.log10(1 + 10000 * spectrogram[i][j])
    return spectrogram


def shuffle_segments(instrument_segments):
    for instr_class in instrument_segments.items():
        random.shuffle(instr_class[1])
    return instrument_segments


def filter_zero_segments(instrument_segments):
    SEGMENT_MAX_ZERO_THRESHOLD = int(settings.SEGMENT_LENGTH * settings.MEL_DATA_POINTS * 0.9)
    clean_instrument_segments = dict()
    for instrument, segments in instrument_segments.items():
        clean_instrument_segments[instrument] = []
        segment_idx = 0
        for segment in segments:
            zeros_in_segment = settings.SEGMENT_LENGTH * settings.MEL_DATA_POINTS - np.count_nonzero(segment[0])
            if zeros_in_segment < SEGMENT_MAX_ZERO_THRESHOLD:
                clean_instrument_segments[instrument].append([])
                clean_instrument_segments[instrument][segment_idx].append(segment[0])  # spectrogram
                clean_instrument_segments[instrument][segment_idx].append(segment[1])  # wav_file_path
                clean_instrument_segments[instrument][segment_idx].append(segment[2])  # segment_idx in file
                segment_idx += 1
            else:
                print "segment dropped: instrument {0}, file {1}, segment {2}".format(instrument, segment[1],
                                                                                      segment[2])
    return clean_instrument_segments


def get_number_of_train_test_segments(instrument_segments):
    min_segments = sys.maxint
    for segments in instrument_segments.values():
        min_segments = min(min_segments, len(segments))
    dividable_min_segments = get_next_int_multiplied_by(settings.TRAIN_TEST_RATIO, min_segments)
    number_of_train = int(dividable_min_segments * settings.TRAIN_TEST_RATIO)
    number_of_test = int(dividable_min_segments - number_of_train)
    return number_of_test, number_of_train


# Splits the data into a training- and a test set
def split_sets(instrument_segments):
    number_of_test, number_of_train = get_number_of_train_test_segments(instrument_segments)

    print "\n"
    print "instruments: {0}".format(instrument_segments.keys())
    print "number of train segments for every instrument class: {0}".format(number_of_train)
    print "number of test segments for every instrument class: {0}".format(number_of_test)
    print "total train segments: {0}".format(len(instrument_segments.keys()) * number_of_train)
    print "total test segments: {0}".format(len(instrument_segments.keys()) * number_of_test)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    test_metainfo = []
    y_mapping = dict()

    train_counter, test_counter, instrument_counter = 0, 0, 0

    for instrument, segments in instrument_segments.items():
        train_segments, test_segments = split_array_in_two(segments, number_of_train, number_of_test)

        for segment in train_segments:
            X_train.append([])
            X_train[train_counter].append([])
            X_train[train_counter][0] = segment[0]
            train_counter += 1
        y_train = y_train + ([instrument_counter] * len(train_segments))

        # this line would set y to a label eg. 'bass'
        # y_train = y_train + ([instrument] * len(train_segments))

        for segment in test_segments:
            X_test.append([])
            test_metainfo.append([])
            X_test[test_counter].append([])
            X_test[test_counter][0] = segment[0]  # spectrogram
            test_metainfo[test_counter].append(segment[1])  # wav_file_path
            test_metainfo[test_counter].append(segment[2])  # segment_idx in file
            test_counter += 1
        y_test = y_test + ([instrument_counter] * len(test_segments))

        y_mapping[instrument_counter] = instrument
        instrument_counter += 1

    X_train = np.array(X_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.int32)
    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.int32)

    return X_train, y_train, X_test, y_test, y_mapping, test_metainfo


def split_array_in_two(values, split_at, max_values_for_second_half):
    first_half = values[0:split_at]
    second_half = values[split_at:(split_at + max_values_for_second_half)]
    return first_half, second_half


def get_next_int_multiplied_by(multiplier, start_value):
    current_min_number = start_value
    while True:
        if (multiplier * current_min_number).is_integer():
            break
        else:
            current_min_number -= 1
    return current_min_number


# Returns the signal data and the sampling rate as a tuple
def get_signal_from_file(wav_file):
    return spectrogram_utils.raw_audio_data(wav_file)


# Returns the spectrogram of a signal
def extract_spectrogram(signal, sampling_rate):
    return spectrogram_utils.mel_spectrogram_from_raw_audio(signal, sampling_rate)


def get_instrument_file_paths(root_dir):
    instrument_file_paths = dict()
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            instrument_file_paths[entry] = []
            for instrument_file in os.listdir(entry_path):
                if instrument_file.endswith(".wav"):
                    file_path = os.path.join(entry_path, instrument_file)
                    instrument_file_paths[entry].append(file_path)
    return instrument_file_paths


def main():
    instrument_files = get_instrument_file_paths(DATA_ROOT_DIR)
    instrument_classes_labels = instrument_files.keys()

    print len(instrument_classes_labels), " instrument classes available:"

    print instrument_classes_labels
    print

    X_train, y_train, X_test, y_test, y_mapping, test_metainfo = generate_datasets(instrument_files)
    print "y mapping: {0}".format(y_mapping)

    utils.save_train_set((X_train, y_train, y_mapping), (os.path.join(DATASET_ROOT_DIR, "train_set.bin")))
    utils.save_test_set((X_test, y_test, test_metainfo), (os.path.join(DATASET_ROOT_DIR, "test_set.bin")))

    print "data preparation finished"
    print "datasets generated in " + DATASET_ROOT_DIR


if __name__ == "__main__":
    main()
