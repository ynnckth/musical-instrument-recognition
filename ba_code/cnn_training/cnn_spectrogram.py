import datetime
import os
import numpy as np

from nolearn.lasagne import NeuralNet
from lasagne import layers
from lasagne import nonlinearities
from time import time

from ba_code.cnn_training.ModelSaver import ModelSaver
from ba_code.cnn_training.ResultPrinter import ResultPrinter
from ba_code.cnn_training.TestSetScorer import TestSetScorer
from ba_code.cnn_training.EmailSender import EmailSender
from ba_code.cnn_training.test_batchiterator import TestSegmentBatchIterator
from ba_code.cnn_training.train_batchiterator import TrainSegmentBatchIterator
from ba_code.cnn_training.instrument_train_splitter import InstrumentTrainSplit
from ba_code.utils import utils
from ba_code import settings

DATASET_ROOT_DIR = "/home/ba16-stdm-streit/datasets"
TRAIN_SET_FNAME = "train_set.bin"
TEST_SET_FNAME = "test_set.bin"
RESULTS_PATH = "/home/ba16-stdm-streit/results"


def load_dataset(path_to_train_set, path_to_test_set):
    X_train, y_train, y_mapping = utils.load_train_set(path_to_train_set)
    X_test, y_test, test_metainfo = utils.load_test_set(path_to_test_set)

    return X_train, y_train, X_test, y_test, y_mapping, test_metainfo


def load_dataset_from_config(train_set_path, test_set_path, instruments, set_size):
    X_train, y_train, X_test, y_test, y_mapping, test_metainfo = load_dataset(path_to_train_set=train_set_path,
                                                                              path_to_test_set=test_set_path)

    train_data_per_instrument = len(y_train) / utils.get_num_of_instr(y_train)
    train_data_per_instrument = int(train_data_per_instrument * set_size)

    test_data_per_instrument = len(y_test) / utils.get_num_of_instr(y_test)

    X_train_selected, y_train_selected = [], []
    X_test_selected, y_test_selected, test_metainfo_selected = [], [], []

    y_mapping_selected = dict()

    indices = utils.get_indices_of_instruments(instruments, y_mapping)
    instrument_idx = 0
    for index in indices:
        start_idx_train = index * train_data_per_instrument
        end_idx_train = start_idx_train + train_data_per_instrument

        start_idx_test = index * test_data_per_instrument
        end_idx_test = start_idx_test + test_data_per_instrument

        X_train_selected.extend(X_train[start_idx_train:end_idx_train])
        y_train_selected.extend(len(y_train[start_idx_train:end_idx_train])*[instrument_idx])

        X_test_selected.extend(X_test[start_idx_test:end_idx_test])
        y_test_selected.extend(len(y_test[start_idx_test:end_idx_test])*[instrument_idx])
        test_metainfo_selected.extend(test_metainfo[start_idx_test:end_idx_test])

        y_mapping_selected[instrument_idx] = y_mapping[index]
        instrument_idx += 1

    X_train_selected = np.array(X_train_selected).astype(np.float32)
    y_train_selected = np.array(y_train_selected).astype(np.int32)
    X_test_selected = np.array(X_test_selected).astype(np.float32)
    y_test_selected = np.array(y_test_selected).astype(np.int32)

    return X_train_selected, y_train_selected, X_test_selected, y_test_selected, y_mapping_selected, test_metainfo_selected


def build_cnn(config):
    # Setup the neural network
    net = NeuralNet(
        layers=config['net'],

        # learning rate parameters
        update_learning_rate=0.001,
        update_momentum=0.9,
        regression=False,

        max_epochs=config.get('epochs', 100),
        verbose=1,
    )
    return net


def prepare_output_directory(config):
    now = datetime.datetime.now()
    file_prefix = str(now.hour) + "-" + str(now.minute) + "_" + str(now.day) + "-" + str(now.month) + "-" + str(
        now.year)
    new_dir = RESULTS_PATH
    if config.get('save_plots', False) or config.get('save_model', False):
        new_dir = os.path.join(RESULTS_PATH, config.get('experiment_name', 'default'))
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
    return new_dir, file_prefix


def main(config):
    result_dir, file_prefix = prepare_output_directory(config)

    train_set_path = os.path.join(DATASET_ROOT_DIR, TRAIN_SET_FNAME)
    test_set_path = os.path.join(DATASET_ROOT_DIR, TEST_SET_FNAME)

    X_train, y_train, X_test, y_test, y_mapping, test_metainfo = load_dataset_from_config(train_set_path, test_set_path,
                                                                                          config['instruments'],
                                                                                          config.get('set_size', 1.0))
    print "Dataset loaded."
    print "Used Instruments: {0}".format(y_mapping)

    net = build_cnn(config)
    print "CNN built."

    # set batch iterators
    net.batch_iterator_train = TrainSegmentBatchIterator(batch_size=settings.MINI_BATCH_SIZE) # used for training
    net.batch_iterator_test = TestSegmentBatchIterator(batch_size=settings.MINI_BATCH_SIZE) # used for validation and test
    net.train_split = InstrumentTrainSplit(eval_size=settings.VAL_TRAIN_RATIO)

    # set result printer
    net.on_training_finished.append(ResultPrinter(config, result_dir, file_prefix))
    net.on_training_finished.append(TestSetScorer(X_test, y_test, test_metainfo, y_mapping, config, result_dir, file_prefix))
    net.on_training_finished.append(ModelSaver(config, result_dir, file_prefix, y_mapping))
    net.on_training_finished.append(EmailSender(config))

    # train the net
    print "Training net..."
    t0 = time()
    net.fit(X_train, y_train)
    print('Finished training. Took %i seconds' % (time() - t0))


if __name__ == "__main__":
    default_config = {
        'net': [
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
            (layers.DenseLayer, {'nonlinearity': nonlinearities.softmax})
        ],
        'epochs': 2000,
        'instruments': ['vocals_male', 'acoustic_guitar', 'electric_guitar', 'overhead', 'vocals_female', 'bass'],
        'save_plots': True,
        'save_model': False,
        'experiment_name': 'default_config',
        'set_size': 1.0
    }
    default_config['net'][len(default_config['net']) - 1][1]['num_units'] = len(default_config['instruments'])
    main(default_config)
