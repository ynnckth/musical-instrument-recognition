import os
import numpy as np
import matplotlib
from itertools import product
# line is needed when plotting to a file
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ba_code.utils.utils import load_from_pickle

NET_ROOT_DIR = "/home/ba16-stdm-streit/results/filter_in_both_small"
NET_FILE_NAME = "0-10_3-6-2016_net.pickle"


def load_net():
    net_path = os.path.join(NET_ROOT_DIR, NET_FILE_NAME)
    net, net_y_mapping = load_from_pickle(net_path)
    return net, net_y_mapping


def my_plot_conv_weights(layer, figsize=(6, 6)):
    W = layer.W.get_value()
    shape = W.shape
    nrows = np.ceil(np.sqrt(shape[0])).astype(int)
    ncols = nrows

    for feature_map in range(shape[1]):
        figs, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
            if i >= shape[0]:
                break
            axes[r, c].imshow(W[i, feature_map], cmap='gray',
                              interpolation='none')
    return plt


def main():
    net, net_y_mapping = load_net()

    my_plot_conv_weights(net.layers_[3], figsize=(4, 4))
    plt.savefig("./results/conv_weights_second_layer_4.png")


if __name__ == "__main__":
    main()
