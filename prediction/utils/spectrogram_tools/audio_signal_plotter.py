
# Use this script to plot the waveform of a wav-file
#
# Librosa does not support 24 bit per sample.
# Only 8, 16 and 32 bit wave uploads are supported.
#
# Usage: audio_signal_plotter xxx.wav

import ntpath
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
from ba_code.prediction import settings

from prediction.utils import mel_spectrogram_from_raw_audio

arguments = sys.argv
file_name = ""


if len(arguments) == 1:
    exit("usage: audio_signal_plotter AUDIO-FILE")
elif len(arguments) == 2:
    file_name = arguments[1]


def plot_signal(signal):
    plt.figure(1)
    signal_plot = plt.subplot(211)
    signal_plot.grid()
    plt.plot(range(0, len(signal)), signal, label="input signal")
    plt.legend(loc='best')
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('signal', fontsize=12)


def plot_spectrogram(spectrogram):
    spect_plot = plt.subplot(212)
    spect_plot.grid()
    librosa.display.specshow(librosa.logamplitude(spectrogram, ref_power=np.max),
                             y_axis='mel',
                             x_axis='time',
                             sr=22050,
                             hop_length=settings.HOP_LENGTH,
                             )

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()


def main():
    signal, sampling_rate = librosa.load(file_name)
    melspectrogram = mel_spectrogram_from_raw_audio(signal, sampling_rate)

    plot_signal(signal)
    plot_spectrogram(melspectrogram)
    plt.suptitle(ntpath.basename(file_name))
    plt.show(True)


if __name__ == "__main__":
    main()