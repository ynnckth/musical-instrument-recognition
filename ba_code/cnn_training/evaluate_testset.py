import os
import sys
import random
import subprocess
from datetime import datetime, timedelta

from pydub import AudioSegment
from pydub.silence import split_on_silence
from sklearn.metrics import accuracy_score

import numpy as np

from ba_code import settings
from ba_code.utils.spectrogram_tools import spectrogram_utils
from ba_code.utils.utils import load_from_pickle
from ba_code.utils import utils

DATA_ROOT_DIR = "/home/ba16-stdm-streit/testset"
NET_ROOT_DIR = "/home/ba16-stdm-streit/results/full_set_seg_5_model/"
NET_FILE_NAME = "11-51_2-6-2016_net.pickle"


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


def remove_silence(instrument_files):
    print "start removing silence"
    for instrument, file_paths in instrument_files.items():
        for file_path in file_paths:
            # remove silence
            audio_signal = AudioSegment.from_file(file_path, format="wav")
            chunks = split_on_silence(audio_signal, min_silence_len=1000, silence_thresh=-55)
            without_silence = chunks[0]
            for i in range(1, len(chunks)):
                without_silence = without_silence + chunks[i]

            without_silence.export(file_path, format("wav"))
    print "removing silence finished"


def normalize_volume(instrument_files):
    print "start normalizing volume"
    for instrument, file_paths in instrument_files.items():
        for file_path in file_paths:
            path_without_ext, extension = os.path.splitext(file_path)
            original_filename = os.path.basename(path_without_ext)
            temp_filename = original_filename + "_topkek" + extension
            temp_path = os.path.join(os.path.dirname(file_path), temp_filename)
            subprocess.check_output(['sox', '--norm=-5', file_path, temp_path])
            os.rename(temp_path, file_path)
    print "normalizing volume finished"


def get_spectrograms_from_filepaths(wav_file_paths, instrument_class):
    instrument_segments = []
    file_counter = 0
    list_idx = 0
    for wav_file_path in wav_file_paths:
        signal, sampling_rate = spectrogram_utils.raw_audio_data(wav_file_path)
        # split wav_file into 10 second segments
        segments = segment_signal(signal, sampling_rate, settings.SEGMENT_SECONDS)
        for segment_idx in range(len(segments)):
            # for each segment generate spectrogram
            spectrogram = spectrogram_utils.mel_spectrogram_from_raw_audio(segments[segment_idx], sampling_rate)
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


# Splits a signal from a wav file into segments of the specified length
def segment_signal(signal, sampling_rate, segmentation_length):
    signal_duration = len(signal) / float(sampling_rate)  # in seconds
    num_of_segments = int(np.ceil(signal_duration / segmentation_length))
    samples_per_segment = segmentation_length * sampling_rate

    segments = []

    for i in range(num_of_segments):
        if i == num_of_segments - 1:
            last_segment = spectrogram_utils.pad_segment_with_zeros(signal[i * samples_per_segment:],
                                                                    samples_per_segment)
            segments.append(last_segment)
        else:
            segments.append(signal[i * samples_per_segment:(i + 1) * samples_per_segment])

    return segments


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
                clean_instrument_segments[instrument][segment_idx].append(segment[0])
                clean_instrument_segments[instrument][segment_idx].append(segment[1])
                clean_instrument_segments[instrument][segment_idx].append(segment[2])
                segment_idx += 1
            else:
                print "segment dropped: instrument {0}, file {1}, segment {2}".format(instrument, segment[1],
                                                                                      segment[2])
    return clean_instrument_segments


def create_segments(instrument_files):
    print "start creating segments"
    instrument_segments = dict()

    for instrument_class, wav_file_paths in instrument_files.items():
        print "start extracting segments for {0}".format(instrument_class)
        segments = get_spectrograms_from_filepaths(wav_file_paths, instrument_class)
        print "finished extracting {0}".format(instrument_class)
        instrument_segments[instrument_class] = segments

    print
    instrument_segments = shuffle_segments(instrument_segments)
    instrument_segments = filter_zero_segments(instrument_segments)
    print "creating segments finished"
    return instrument_segments


def create_X_and_y(instrument_segments):
    X_test = []
    y_test = []
    test_metainfo = []
    y_mapping = dict()

    test_counter, instrument_counter = 0, 0

    for instrument, segments in instrument_segments.items():
        for segment in segments:
            X_test.append([])
            test_metainfo.append([])
            X_test[test_counter].append([])
            X_test[test_counter][0] = segment[0]  # spectrogram
            test_metainfo[test_counter].append(segment[1])  # wav_file_path
            test_metainfo[test_counter].append(segment[2])  # segment_idx in file
            test_counter += 1
        y_test = y_test + ([instrument_counter] * len(segments))

        y_mapping[instrument_counter] = instrument
        instrument_counter += 1

    X_test = np.array(X_test).astype(np.float32)
    y_test = np.array(y_test).astype(np.int32)

    return X_test, y_test, y_mapping, test_metainfo


def load_net():
    net_path = os.path.join(NET_ROOT_DIR, NET_FILE_NAME)
    net, net_y_mapping = load_from_pickle(net_path)
    return net, net_y_mapping


def get_fixed_y(net_y_mapping, y_mapping, yb):
    reverse_dict = dict((v, k) for k, v in net_y_mapping.items())
    for instr_key, instrument in y_mapping.items():
        key_in_net = reverse_dict[instrument]
        for idx in range(len(yb)):
            if yb[idx] == instr_key:
                yb[idx] = key_in_net
    return yb


def main():
    instrument_files = get_instrument_file_paths(DATA_ROOT_DIR)
    instrument_classes_labels = instrument_files.keys()

    print len(instrument_classes_labels), " instrument classes available:"

    print instrument_classes_labels
    print

    net, net_y_mapping = load_net()

    prep_data = raw_input("Prepare data for net? [y/n] (remove silence and normalization) This will change the uploads"
                          " PERMANTENTLY!!!: ")
    if prep_data == 'Y' or prep_data == 'y':
        normalize_volume(instrument_files)
        remove_silence(instrument_files)

    instrument_segments = create_segments(instrument_files)
    X, y, y_mapping, metainfo = create_X_and_y(instrument_segments)

    Xb, yb, metainfo = utils.extract_mini_segments_with_metainfo(X, y, metainfo)
    Xb, yb, metainfo = utils.filter_mini_segments_with_metainfo(Xb, yb, metainfo)

    yb = get_fixed_y(net_y_mapping, y_mapping, yb)

    predictions = net.predict(Xb)
    proba_predictions = net.predict_proba(Xb)

    score = float(accuracy_score(predictions, yb))

    print "Score on testset is: {0}\n".format(str(score))

    for i in range(len(predictions)):
        if predictions[i] != yb[i]:
            dt = datetime.now()
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
            dt = dt + timedelta(seconds=(metainfo[i][2] * settings.MINI_BATCH_SECONDS))
            float_formatter = lambda x: "%.2f" % x
            np.set_printoptions(formatter={'float_kind': float_formatter})
            np.set_printoptions(precision=2)
            instrument_proba = dict()
            for instr_idx in range(len(proba_predictions[i])):
                instrument_proba[net_y_mapping[instr_idx]] = "%.2f" % proba_predictions[i][instr_idx]
            output = "{0} classified as {1}; look at {2}:{3}; \n\tprediction: {4}".format(metainfo[i][0][28:],
                                                                                        net_y_mapping[predictions[i]],
                                                                                        dt.minute, dt.second,
                                                                                        instrument_proba)
            print output

if __name__ == "__main__":
    main()
