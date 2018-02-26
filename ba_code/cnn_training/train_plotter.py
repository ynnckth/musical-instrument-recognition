
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

from ba_code.utils import utils


def interpolate(accuracies, train_losses, valid_losses):
    x_acc = np.linspace(0, len(accuracies), num=len(accuracies), endpoint=False)
    spl_acc = UnivariateSpline(x_acc, accuracies)
    spl_acc.set_smoothing_factor(0.005)

    x_train_loss = np.linspace(0, len(train_losses), num=len(train_losses), endpoint=False)
    spl_train = UnivariateSpline(x_train_loss, train_losses)
    spl_train.set_smoothing_factor(0.08)

    x_valid_loss = np.linspace(0, len(valid_losses), num=len(valid_losses), endpoint=False)
    spl_valid = UnivariateSpline(x_valid_loss, valid_losses)
    spl_valid.set_smoothing_factor(0.08)

    return x_acc, spl_acc, x_train_loss, spl_train, x_valid_loss, spl_valid

train_history = utils.load_from_pickle("/home/b4nsh33/10-46_18-5-2016_train_history.pickle")

train_loss = []
valid_loss = []
valid_accuracy = []

for i in range(len(train_history)):
    train_loss.append(train_history[i]['train_loss'])
    valid_loss.append(train_history[i]['valid_loss'])
    valid_accuracy.append(train_history[i]['valid_accuracy'])

x_acc, spl_acc, x_train_loss, spl_train, x_valid_loss, spl_valid = \
    interpolate(valid_accuracy, train_loss, valid_loss)


def plot_train_validation_loss():
    plt.subplot(211)
    plt.semilogy(x_train_loss, spl_train(x_acc), label='Training-Loss', color='b')
    plt.semilogy(train_loss, alpha=0.2, color='b')

    plt.semilogy(x_valid_loss, spl_valid(x_acc), label='Validation-Loss', color='g')
    plt.semilogy(valid_loss, alpha=0.2, color='g')

    plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    plt.grid()
    plt.legend(loc='best', shadow=True)


def plot_validation_accuracy():
    plt.subplot(212)
    plt.plot(x_acc, spl_acc(x_acc), label='Accuracy', color='b')
    plt.plot(valid_accuracy, alpha=0.2, color='b')

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.grid()
    plt.legend(loc='best', shadow=True)


def main():
    plt.figure(1)
    plot_train_validation_loss()
    plot_validation_accuracy()
    plt.show(True)


main()






'''
file_name = self.file_prefix + '_loss.png'
path_to_saved_file = os.path.join(self.result_dir, file_name)
plt.savefig(path_to_saved_file)
plt.clf()
'''

'''
file_name = self.file_prefix + '_acc.png'
path_to_saved_file = os.path.join(self.result_dir, file_name)
plt.savefig(path_to_saved_file)
plt.clf()
'''