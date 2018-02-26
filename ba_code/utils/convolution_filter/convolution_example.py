from PIL import Image
import numpy as np
from scipy import signal


def img_to_array(fname):
    return np.asarray(Image.open(fname), dtype=np.float32)


def save_as_img(img_as_array, fname):
    Image.fromarray(img_as_array.round().astype(np.uint8)).save(fname)


def norm(img_as_array):
    return 255.*np.absolute(img_as_array) / np.max(img_as_array)

img = img_to_array("img.jpg")

# the kernel is the filter (the sliding window) of the convolution
kernel_filter = [[-1., -1., -1], [-1., 8., -1], [-1., -1., -1.]]
save_as_img(norm(signal.convolve(img.mean(axis=-1), kernel_filter)), 'output.jpg')  # img.mean(axis=-1) converts image
                                                                                    # to black/white
