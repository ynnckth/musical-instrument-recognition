import subprocess

import os
import numpy as np

from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
from pydub import AudioSegment
from pydub.silence import split_on_silence

from ba_code import settings
from ba_code.cnn_training.test_batchiterator import TestSegmentBatchIterator
from ba_code.utils import utils
from ba_code.utils.spectrogram_tools import spectrogram_utils
from ba_code.utils.utils import load_from_pickle


CNN_WEIGHTS_PATH = os.path.normpath(os.path.join(os.getcwd(), "../trained_models/full_set_model_5000/23-15_1-6-2016_net_weights.pickle"))
CNN_Y_MAPPING_PATH = os.path.normpath(os.path.join(os.getcwd(), "../trained_models/full_set_model_5000/23-15_1-6-2016_y_mapping.pickle"))

global_cnn = None
global_y_mapping = None


def normalize_volume(audio_file_path):
    print "normalizing volume"
    path_without_ext, extension = os.path.splitext(audio_file_path)
    original_filename = os.path.basename(path_without_ext)
    tmp_normalized_path = os.path.join(os.path.dirname(audio_file_path), original_filename + "_tmp" + extension)
    subprocess.check_output(['/usr/local/bin/sox', '--norm=-5', audio_file_path, tmp_normalized_path])
    os.rename(tmp_normalized_path, audio_file_path)


def remove_silence(audio_file_path):
    print "removing silence"
    audio_signal = AudioSegment.from_file(audio_file_path, format="wav")
    audio_chunks = split_on_silence(audio_signal, min_silence_len=1000, silence_thresh=-55)
    audio_without_silence = audio_chunks[0]
    for i in range(1, len(audio_chunks)):
        audio_without_silence = audio_without_silence + audio_chunks[i]

    audio_without_silence.export(audio_file_path, format("wav"))


def create_segments(audio_file_path):
    print "creating segments"
    instrument_segments = get_spectrograms_from_filepath(audio_file_path)
    instrument_segments = filter_zero_segments(instrument_segments)
    return instrument_segments


def get_spectrograms_from_filepath(wav_file_path):
    instrument_segments = []
    list_idx = 0

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
    return instrument_segments


def dynamic_compression(spectrogram):
    for i in range(len(spectrogram)):
        for j in range(len(spectrogram[i])):
            spectrogram[i][j] = np.log10(1 + 10000 * spectrogram[i][j])
    return spectrogram


def filter_zero_segments(instrument_segments):
    SEGMENT_MAX_ZERO_THRESHOLD = int(settings.SEGMENT_LENGTH * settings.MEL_DATA_POINTS * 0.9)
    clean_instrument_segments = []
    segment_idx = 0
    for segment in instrument_segments:
        zeros_in_segment = settings.SEGMENT_LENGTH * settings.MEL_DATA_POINTS - np.count_nonzero(segment[0])
        if zeros_in_segment < SEGMENT_MAX_ZERO_THRESHOLD:
            clean_instrument_segments.append([])
            clean_instrument_segments[segment_idx].append(segment[0])
            clean_instrument_segments[segment_idx].append(segment[1])
            clean_instrument_segments[segment_idx].append(segment[2])
            segment_idx += 1
        else:
            print "segment dropped: file {0}, segment {1}".format(segment[1], segment[2])
    return clean_instrument_segments


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


def create_X_and_y(instrument_segments):
    X_test = []
    y_test = []
    test_metainfo = []

    test_counter = 0

    for segment in instrument_segments:
        X_test.append([])
        test_metainfo.append([])
        X_test[test_counter].append([])
        X_test[test_counter][0] = segment[0]  # spectrogram
        test_metainfo[test_counter].append(segment[1])  # wav_file_path
        test_metainfo[test_counter].append(segment[2])  # segment_idx in file
        test_counter += 1

    X_test = np.array(X_test).astype(np.float32)
    y_test = np.zeros(len(X_test), dtype=np.int32)

    return X_test, y_test, test_metainfo


def load_net():
    net = NeuralNet(
        layers=[
            # input layer
            (layers.InputLayer,
             {'shape': (None, settings.CHANNELS, settings.MINI_SEGMENT_LENGTH, settings.MEL_DATA_POINTS)}),

            # convolution layers 1
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (8, 1)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 1), 'stride': (2, 1)}),

            # convolution layers 2
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (8, 1)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 1), 'stride': (2, 1)}),

            # dense layer
            (layers.DenseLayer, {'num_units': 100}),
            (layers.DropoutLayer, {}),
            (layers.DenseLayer, {'num_units': 50}),

            # output layer
            (layers.DenseLayer, {'num_units': 6, 'nonlinearity': nonlinearities.softmax})
        ],

        # learning rate parameters
        update_learning_rate=0.001,
        update_momentum=0.9,
        regression=False,

        max_epochs=999,
        verbose=1,
    )

    net.batch_iterator_test = TestSegmentBatchIterator(batch_size=settings.MINI_BATCH_SIZE)
    y_mapping = load_from_pickle(CNN_Y_MAPPING_PATH)
    net.load_params_from(CNN_WEIGHTS_PATH)

    return net, y_mapping


def initialize_cnn():
    print('Loading cnn model from %s' % CNN_WEIGHTS_PATH)
    cnn, y_mapping = load_net()

    global global_cnn, global_y_mapping
    global_cnn = cnn
    global_y_mapping = y_mapping


def predict_instrument(file_path):
    print('File preparation')

    normalize_volume(file_path)
    remove_silence(file_path)

    instrument_segments = create_segments(file_path)
    X, y, metainfo = create_X_and_y(instrument_segments)

    Xb, yb, metainfo = utils.extract_mini_segments_with_metainfo(X, y, metainfo)
    Xb, yb, metainfo = utils.filter_mini_segments_with_metainfo(Xb, yb, metainfo)

    print('Predicting...')

    proba_predictions = global_cnn.predict_proba(Xb)

    predictions = dict()
    proba_mean = np.mean(proba_predictions, axis=0)
    for instrument_idx, instrument_name in global_y_mapping.items():
        predictions[instrument_name] = round(proba_mean[instrument_idx], 3)

    classified_instrument = global_y_mapping[np.argmax(proba_mean)]
    score = proba_mean[np.argmax(proba_mean)]

    return predictions, classified_instrument, score


def main(file_path):
    print('Loading model from %s' % CNN_WEIGHTS_PATH)
    cnn, y_mapping = load_net()

    print('Preparing audio file...')

    normalize_volume(file_path)
    remove_silence(file_path)

    instrument_segments = create_segments(file_path)
    X, y, metainfo = create_X_and_y(instrument_segments)

    Xb, yb, metainfo = utils.extract_mini_segments_with_metainfo(X, y, metainfo)
    Xb, yb, metainfo = utils.filter_mini_segments_with_metainfo(Xb, yb, metainfo)

    print('Predicting...')

    predictions = cnn.predict(Xb)
    proba_predictions = cnn.predict_proba(Xb)

    return predictions, proba_predictions


if __name__ == "__main__":
    main("PATH_TO_WAV_FILE")
