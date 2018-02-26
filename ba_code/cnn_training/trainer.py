from lasagne import layers

from ba_code import settings
from ba_code.cnn_training import cnn_spectrogram as cnn
from lasagne import nonlinearities

# available instruments:
# ['vocals_male', 'acoustic_guitar', 'electric_guitar', 'overhead', 'vocals_female', 'bass']

configs = [
    {
        'net': [
            # input layer
            (layers.InputLayer,
             {'shape': (None, settings.CHANNELS, settings.MINI_SEGMENT_LENGTH, settings.MEL_DATA_POINTS)}),

            # convolution layers 1
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (1, 8)}),
            (layers.MaxPool2DLayer, {'pool_size': (1, 4), 'stride': (1, 2)}),

            # convolution layers 2
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (1, 8)}),
            (layers.MaxPool2DLayer, {'pool_size': (1, 4), 'stride': (1, 2)}),

            # dense layer
            (layers.DenseLayer, {'num_units': 100}),
            (layers.DropoutLayer, {}),
            (layers.DenseLayer, {'num_units': 50}),

            # output layer
            (layers.DenseLayer, {'nonlinearity': nonlinearities.softmax})
        ],
        'epochs': 5000,
        'instruments': ['vocals_male', 'acoustic_guitar', 'electric_guitar', 'overhead', 'vocals_female', 'bass'],
        'save_plots': True,
        'save_model': True,
        'experiment_name': 'filter_in_freq'
    },

    {
        'net': [
            # input layer
            (layers.InputLayer,
             {'shape': (None, settings.CHANNELS, settings.MINI_SEGMENT_LENGTH, settings.MEL_DATA_POINTS)}),

            # convolution layers 1
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (4, 4)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 4), 'stride': (2, 2)}),

            # convolution layers 2
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (4, 4)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 4), 'stride': (2, 2)}),

            # dense layer
            (layers.DenseLayer, {'num_units': 100}),
            (layers.DropoutLayer, {}),
            (layers.DenseLayer, {'num_units': 50}),

            # output layer
            (layers.DenseLayer, {'nonlinearity': nonlinearities.softmax})
        ],
        'epochs': 5000,
        'instruments': ['vocals_male', 'acoustic_guitar', 'electric_guitar', 'overhead', 'vocals_female', 'bass'],
        'save_plots': True,
        'save_model': True,
        'experiment_name': 'filter_in_both_small'
    },

    {
        'net': [
            # input layer
            (layers.InputLayer,
             {'shape': (None, settings.CHANNELS, settings.MINI_SEGMENT_LENGTH, settings.MEL_DATA_POINTS)}),

            # convolution layers 1
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (8, 8)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 4), 'stride': (2, 2)}),

            # convolution layers 2
            (layers.Conv2DLayer, {'num_filters': 32, 'filter_size': (8, 8)}),
            (layers.MaxPool2DLayer, {'pool_size': (4, 4), 'stride': (2, 2)}),

            # dense layer
            (layers.DenseLayer, {'num_units': 100}),
            (layers.DropoutLayer, {}),
            (layers.DenseLayer, {'num_units': 50}),

            # output layer
            (layers.DenseLayer, {'nonlinearity': nonlinearities.softmax})
        ],
        'epochs': 5000,
        'instruments': ['vocals_male', 'acoustic_guitar', 'electric_guitar', 'overhead', 'vocals_female', 'bass'],
        'save_plots': True,
        'save_model': True,
        'experiment_name': 'filter_in_both_big'
    }
]


def main():
    for config in configs:
        config['net'][len(config['net'])-1][1]['num_units'] = len(config['instruments'])
        cnn.main(config)


if __name__ == '__main__':
    main()
