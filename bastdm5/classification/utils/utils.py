import cPickle as pickle
import math

import numpy as np
from scipy.interpolate import UnivariateSpline

from bastdm5.classification import settings


def save_train_set(train_set, path):
    with open(path, 'wb') as f:
        np.save(f, train_set[0])
        np.save(f, train_set[1])
        np.save(f, train_set[2])


def save_test_set(test_set, path):
    with open(path, 'wb') as f:
        np.save(f, test_set[0])
        np.save(f, test_set[1])
        np.save(f, test_set[2])


def load_train_set(path):
    with open(path, 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)
        y_mapping = np.load(f).item()
    return X_train, y_train, y_mapping


def load_test_set(path):
    with open(path, 'rb') as f:
        X_test = np.load(f)
        y_test = np.load(f)
        test_metainfo = np.load(f)
    return X_test, y_test, test_metainfo


def save_as_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, -1)


def load_from_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_num_of_instr(y):
    num_instr = 1
    curr_instr = y[0]
    for i in range(len(y)):
        if curr_instr != y[i]:
            num_instr += 1
            curr_instr = y[i]
    return num_instr


def get_indices_of_instruments(instruments_label, y_mapping):
    inverse_y_mapping = {v: k for k, v in y_mapping.items()}
    instrument_indices = []
    for instrument in instruments_label:
        if instrument not in y_mapping.values():
            raise ValueError("Instrument: '{0}' does not exist in dataset".format(instrument))
        instrument_indices.append(inverse_y_mapping[instrument])
    instrument_indices.sort()
    return instrument_indices


def extract_mini_segments(X_raw, y_raw):
    new_segment_length = len(X_raw) * settings.MINI_SEGMENTS_IN_SEGMENT

    Xb = np.zeros((new_segment_length, 1, settings.MINI_SEGMENT_LENGTH, settings.MEL_DATA_POINTS), dtype=np.float32)
    yb = np.zeros(new_segment_length, dtype=np.int32)

    for i in range(0, len(X_raw)):
        segments = split_spectrogram(X_raw[i, 0])

        start_idx = i * settings.MINI_SEGMENTS_IN_SEGMENT
        end_idx = start_idx + settings.MINI_SEGMENTS_IN_SEGMENT

        Xb[start_idx: end_idx, 0] = segments
        yb[start_idx: end_idx] = [y_raw[i]] * settings.MINI_SEGMENTS_IN_SEGMENT

    return Xb, yb


def extract_mini_segments_with_metainfo(X_raw, y_raw, metainfo):
    new_segment_length = len(X_raw) * settings.MINI_SEGMENTS_IN_SEGMENT

    Xb = np.zeros((new_segment_length, 1, settings.MINI_SEGMENT_LENGTH, settings.MEL_DATA_POINTS), dtype=np.float32)
    yb = np.zeros(new_segment_length, dtype=np.int32)
    new_metainfo = []

    mini_segmend_idx = 0
    for i in range(0, len(X_raw)):
        segments = split_spectrogram(X_raw[i, 0])

        start_idx = i * settings.MINI_SEGMENTS_IN_SEGMENT
        end_idx = start_idx + settings.MINI_SEGMENTS_IN_SEGMENT

        Xb[start_idx: end_idx, 0] = segments
        yb[start_idx: end_idx] = [y_raw[i]] * settings.MINI_SEGMENTS_IN_SEGMENT

        segment_idx = int(metainfo[i][1])
        start_mini_segment_idx = segment_idx * settings.MINI_SEGMENTS_IN_SEGMENT
        end_mini_segment_idx = start_mini_segment_idx + settings.MINI_SEGMENTS_IN_SEGMENT

        new_metainfo.extend([metainfo[i]] * settings.MINI_SEGMENTS_IN_SEGMENT)
        for j in range(start_mini_segment_idx, end_mini_segment_idx):
            if isinstance(new_metainfo[mini_segmend_idx], np.ndarray):
                temp_list = new_metainfo[mini_segmend_idx].tolist()
            else:
                temp_list = new_metainfo[mini_segmend_idx][:]
            temp_list[1] = int(temp_list[1])
            temp_list.append(j)
            new_metainfo[mini_segmend_idx] = temp_list
            mini_segmend_idx += 1

    return Xb, yb, new_metainfo


def filter_mini_segments(X, y):
    # threshold = max number of zeros tolerated
    MINI_SEGMENT_ZERO_THRESHOLD = int(settings.MINI_SEGMENT_LENGTH * settings.MEL_DATA_POINTS * 0.25)

    Xb, yb = [], []
    idx = 0
    for i in range(len(X)):
        mini_segment = X[i, 0]
        if get_zeros_in_segment(mini_segment) < MINI_SEGMENT_ZERO_THRESHOLD:
            # segment is good
            Xb.append([])
            Xb[idx].append([])
            Xb[idx][0] = mini_segment
            yb.append(y[i])
            idx += 1

    Xb = np.array(Xb).astype(np.float32)
    yb = np.array(yb).astype(np.int32)
    return Xb, yb


def filter_mini_segments_with_metainfo(X, y, metainfo):
    # threshold = max number of zeros tolerated
    MINI_SEGMENT_ZERO_THRESHOLD = int(settings.MINI_SEGMENT_LENGTH * settings.MEL_DATA_POINTS * 0.25)

    Xb, yb, new_metainfo = [], [], []
    idx = 0
    for i in range(len(X)):
        mini_segment = X[i, 0]
        if get_zeros_in_segment(mini_segment) < MINI_SEGMENT_ZERO_THRESHOLD:
            # segment is good
            Xb.append([])
            Xb[idx].append([])
            Xb[idx][0] = mini_segment
            yb.append(y[i])
            new_metainfo.append(metainfo[i])
            idx += 1

    Xb = np.array(Xb).astype(np.float32)
    yb = np.array(yb).astype(np.int32)
    return Xb, yb, new_metainfo


def split_spectrogram(spectrogram):
    segments = []
    for i in range(settings.MINI_SEGMENTS_IN_SEGMENT):
        start = i * settings.MINI_SEGMENT_LENGTH
        end = (i + 1) * settings.MINI_SEGMENT_LENGTH

        segments.append(spectrogram[start:end])
    return segments


def get_zeros_in_segment(segment):
    return settings.MINI_SEGMENT_LENGTH * settings.MEL_DATA_POINTS - np.count_nonzero(segment)


def interpolate(accuracies, train_losses, valid_losses):
    x_acc = np.linspace(0, len(accuracies), num=len(accuracies), endpoint=False)
    spl_acc = UnivariateSpline(x_acc, accuracies)
    acc_smoothing_factor = (len(accuracies) - math.sqrt(2 * len(accuracies))) * np.std(accuracies) ** 2
    spl_acc.set_smoothing_factor(acc_smoothing_factor)

    x_train_loss = np.linspace(0, len(train_losses), num=len(train_losses), endpoint=False)
    spl_train = UnivariateSpline(x_train_loss, train_losses)
    train_loss_smoothing_factor = (len(x_train_loss) - math.sqrt(2 * len(x_train_loss))) * np.std(x_train_loss) ** 2
    spl_train.set_smoothing_factor(train_loss_smoothing_factor)

    x_valid_loss = np.linspace(0, len(valid_losses), num=len(valid_losses), endpoint=False)
    spl_valid = UnivariateSpline(x_valid_loss, valid_losses)
    valid_loss_smoothing_factor = (len(x_valid_loss) - math.sqrt(2 * len(x_valid_loss))) * np.std(x_valid_loss) ** 2
    spl_valid.set_smoothing_factor(valid_loss_smoothing_factor)

    return x_acc, spl_acc, x_train_loss, spl_train, x_valid_loss, spl_valid
