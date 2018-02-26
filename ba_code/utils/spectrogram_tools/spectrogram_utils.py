
import librosa
import numpy as np
import matplotlib.pyplot as plt


from ba_code import settings


# Splits a signal from a wav file into segments of the specified length
def segment_signal(signal, sampling_rate, segmentation_length):
    signal_duration = len(signal) / float(sampling_rate)  # in seconds
    num_of_segments = int(np.ceil(signal_duration / segmentation_length))
    samples_per_segment = segmentation_length * sampling_rate

    segments = []

    for i in range(num_of_segments):
        if i == num_of_segments - 1:
            last_segment = pad_segment_with_zeros(signal[i * samples_per_segment:], samples_per_segment)
            segments.append(last_segment)
        else:
            segments.append(signal[i * samples_per_segment:(i + 1) * samples_per_segment])

    return segments


# Appends zeros to the end of a segment to reach the segment size
def pad_segment_with_zeros(segment, segment_size):
    padding = []
    for i in range(len(segment), segment_size):
        padding.append(0)

    return np.concatenate((segment, padding), axis=0)


def mel_spectrogram_from_wav_file(wav_file):
    signal, sampling_rate = raw_audio_data(wav_file)
    print "Extracting spectrogram from ", wav_file, "..."
    melspectrogram = librosa.feature.melspectrogram(y=signal, sr=sampling_rate, n_mels=128,
                                                    n_fft=settings.WINDOW_LENGTH, hop_length=settings.HOP_LENGTH,
                                                    fmax=settings.FMAX)

    return melspectrogram


def mel_spectrogram_from_raw_audio(signal, sampling_rate):
    melspectrogram = librosa.feature.melspectrogram(y=signal, sr=sampling_rate, n_mels=settings.MEL_DATA_POINTS,
                                                    n_fft=settings.WINDOW_LENGTH, hop_length=settings.HOP_LENGTH,
                                                    fmax=settings.FMAX)
    return melspectrogram


def raw_audio_data(wav_file):
    signal, sampling_rate = librosa.load(wav_file)
    return signal, sampling_rate


def plot_spectrogram(spectrogram):
    librosa.display.specshow(librosa.logamplitude(spectrogram, ref_power=np.max),
        y_axis='mel', fmax=settings.FMAX,
        x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('mel spectrogram')
    plt.tight_layout()
    plt.show(True)


def save_spectrogram_plot(spectrogram, file_path):
    plt.figure(1)
    librosa.display.specshow(librosa.logamplitude(spectrogram),
                             y_axis='mel', fmax=settings.FMAX,
                             x_axis='time', hop_length=settings.HOP_LENGTH, sr=settings.SAMPLING_RATE)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()


def save_signal_plot(signal, file_path):
    plt.figure(1)
    plt.plot(range(0, len(signal)), signal, label="input signal")
    plt.legend(loc='best')
    plt.xlabel('sample', fontsize=12)
    plt.ylabel('signal', fontsize=12)
    plt.savefig(file_path)
    plt.clf()

def dyn_range_compression(x):
    return np.log10(1 + 10000 * x)