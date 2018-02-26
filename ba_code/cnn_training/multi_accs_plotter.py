import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np

from ba_code.utils import utils


def interpolate(accuracies, smoothing):
    x_acc = np.linspace(0, len(accuracies), num=len(accuracies), endpoint=False)
    spl_acc = UnivariateSpline(x_acc, accuracies)
    spl_acc.set_smoothing_factor(smoothing)

    return x_acc, spl_acc


picklesPaths = dict()

picklesPaths["/home/nutella/BA/repositories/ba-code/results/full_set_seg_5/9-15_1-6-2016_train_history.pickle"] = [0.2, "8x1, 4x1", 'brown']
picklesPaths["/home/nutella/BA/repositories/ba-code/results/filter_in_freq/18-34_2-6-2016_train_history.pickle"] = [0.1, "1x8, 1x4", 'green']
picklesPaths["/home/nutella/BA/repositories/ba-code/results/filter_in_both_small/0-10_3-6-2016_train_history.pickle"] = [0.2, "4x4, 4x4", 'red']
picklesPaths["/home/nutella/BA/repositories/ba-code/results/filter_in_both_big/5-3_3-6-2016_train_history.pickle"] = [0.2, "8x8, 4x4", 'blue']


acc_histories = []
current_history_index = 0

for path, options in picklesPaths.items():
    train_history = utils.load_from_pickle(path)
    experiment_name = path[39:(path[39:].find("/") + 39)]  # unused
    acc_histories.append([])

    valid_accuracy = []
    for i in range(len(train_history)):
        valid_accuracy.append(train_history[i]['valid_accuracy'])
    acc_histories[current_history_index].append(valid_accuracy)
    acc_histories[current_history_index].append(options[0])
    acc_histories[current_history_index].append(options[1])
    acc_histories[current_history_index].append(options[2])
    current_history_index += 1

plt.ylim((0.85, 1))
plt.figure(1)
last_epoch = 2000

for acc_history in acc_histories:
    x_acc, spl_acc = interpolate(acc_history[0][0:last_epoch], acc_history[1])

    plt.plot(x_acc, spl_acc(x_acc), label=acc_history[2], color=acc_history[3])
    plt.plot(acc_history[0][0:last_epoch], alpha=0.15, color='black')

    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')

plt.grid()
plt.legend(loc='best', shadow=True)


#file_name = '/home/nutella/segment_lengths.png'
#plt.savefig(file_name)
plt.show(True)
