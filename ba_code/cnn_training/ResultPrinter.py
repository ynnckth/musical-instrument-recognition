import os

from ba_code.utils import utils

import matplotlib
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import DropoutLayer

# line is needed when plotting to a file
matplotlib.use('Agg')

import matplotlib.pyplot as plt


class ResultPrinter:
    def __init__(self, config, result_dir, file_prefix):
        self.instruments = config['instruments']
        self.save_plots = config.get('save_plots', False)
        self.file_prefix = file_prefix
        self.result_dir = result_dir
        self.experiment_name = config.get('experiment_name', 'default')
        self.set_size = config.get('set_size', 1.0)

    # The call method will be called from nolearn when all epochs trained
    def __call__(self, nn, train_history):
        print
        if self.save_plots:
            self.save_train_valid_plot(train_history)
            self.save_net_informations(nn)
            self.save_train_history(train_history)
            print 'Saving the results to ' + self.result_dir + "/" + self.file_prefix
        else:
            print('Plots not saved')

    def save_train_valid_plot(self, train_history):
        train_loss = [row['train_loss'] for row in train_history]
        valid_loss = [row['valid_loss'] for row in train_history]
        valid_accuracy = [row['valid_accuracy'] for row in train_history]

        x_acc, spl_acc, x_train_loss, spl_train, x_valid_loss, spl_valid = \
            utils.interpolate(valid_accuracy, train_loss, valid_loss)

        plt.figure(1)
        plt.plot(x_train_loss, spl_train(x_acc), label='Training-Loss', color='b')
        plt.plot(train_loss, alpha=0.2, color='b')

        plt.plot(x_valid_loss, spl_valid(x_acc), label='Validation-Loss', color='g')
        plt.plot(valid_loss, alpha=0.2, color='g')
        plt.ylim(0, 2)

        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.grid()
        plt.legend(loc='upper right', shadow=True)

        file_name = self.file_prefix + '_loss.png'
        path_to_saved_file = os.path.join(self.result_dir, file_name)
        plt.savefig(path_to_saved_file)
        plt.clf()

        # ----------------------------------------------------------

        plt.figure(2)
        plt.plot(x_acc, spl_acc(x_acc), label='Accuracy', color='b')
        plt.plot(valid_accuracy, alpha=0.2, color='b')
        plt.ylim(0, 1)

        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.grid()
        plt.legend(loc='lower right', shadow=True)

        file_name = self.file_prefix + '_acc.png'
        path_to_saved_file = os.path.join(self.result_dir, file_name)
        plt.savefig(path_to_saved_file)
        plt.clf()

    def save_net_informations(self, nn):
        file_name = self.file_prefix + '_nn_info.txt'
        path_to_saved_file = os.path.join(self.result_dir, file_name)

        header_output = "\nExperiment on {0}\n".format(self.file_prefix)
        header_output += "-----------------------------------------------------\n\n"

        instrument_output = "Instruments: {0}\n".format(str(self.instruments))

        experiment_output = "Experiment name: {0}\n".format(self.experiment_name)

        epoch_output = "Number of epochs: {0}\n".format(nn.max_epochs)

        set_size_output = "Used Set size: {0}\n\n".format(self.set_size)

        layer_output = "Layer informations\n"
        layer_output += "--------------------------------------\n\n"
        layer_output += ResultPrinter.get_layer_informations(nn)

        info_file = open(path_to_saved_file, 'w')
        info_file.write(header_output)
        info_file.write(instrument_output)
        info_file.write(experiment_output)
        info_file.write(epoch_output)
        info_file.write(set_size_output)
        info_file.write(layer_output)
        info_file.close()

    def save_train_history(self, train_history):
        train_loss = [row['train_loss'] for row in train_history]
        valid_loss = [row['valid_loss'] for row in train_history]
        valid_accuracy = [row['valid_accuracy'] for row in train_history]

        epochs = dict()
        for epoch_idx in range(len(train_history)):
            epochs[epoch_idx] = dict()
            epochs[epoch_idx]['train_loss'] = train_loss[epoch_idx]
            epochs[epoch_idx]['valid_loss'] = valid_loss[epoch_idx]
            epochs[epoch_idx]['valid_accuracy'] = valid_accuracy[epoch_idx]

        file_name = self.file_prefix + '_train_history.pickle'
        path_to_saved_file = os.path.join(self.result_dir, file_name)
        utils.save_as_pickle(epochs, path_to_saved_file)

    @staticmethod
    def get_layer_informations(nn):
        output = ""
        for layer in nn.layers_.values():
            if isinstance(layer, InputLayer):
                output += ResultPrinter.printInputLayerInfos(layer)
            elif isinstance(layer, Conv2DLayer):
                output += ResultPrinter.getConv2DLayerInfos(layer)
            elif isinstance(layer, MaxPool2DLayer):
                output += ResultPrinter.getMaxPool2DLayerInfos(layer)
            elif isinstance(layer, DenseLayer):
                output += ResultPrinter.getDenseLayerInfos(layer)
            elif isinstance(layer, DropoutLayer):
                output += ResultPrinter.getDropoutLayerInfos(layer)
        return output

    @staticmethod
    def printInputLayerInfos(layer):
        infos = ""
        infos += "name: " + layer.name + "\n"
        infos += "shape: " + str(layer.shape) + "\n"
        return infos + "\n"

    @staticmethod
    def getConv2DLayerInfos(layer):
        infos = ""
        infos += "name: " + layer.name + "\n"
        infos += "num_filters: " + str(layer.num_filters) + "\n"
        infos += "filter_size: " + str(layer.filter_size) + "\n"
        return infos + "\n"

    @staticmethod
    def getMaxPool2DLayerInfos(layer):
        infos = ""
        infos += "name: " + layer.name + "\n"
        infos += "pool_size: " + str(layer.pool_size) + "\n"
        infos += "stride: " + str(layer.stride) + "\n"
        return infos + "\n"

    @staticmethod
    def getDenseLayerInfos(layer):
        infos = ""
        infos += "name: " + layer.name + "\n"
        infos += "num_units: " + str(layer.num_units) + "\n"
        return infos + "\n"

    @staticmethod
    def getDropoutLayerInfos(layer):
        infos = ""
        infos += "name: " + layer.name + "\n"
        infos += "p: " + str(layer.p) + "\n"
        return infos + "\n"
