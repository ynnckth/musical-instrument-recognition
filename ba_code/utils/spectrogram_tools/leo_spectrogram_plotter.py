import matplotlib.pyplot as plt
import librosa

from ba_code import settings
from ba_code.utils.spectrogram_tools.spectrogram_utils import mel_spectrogram_from_raw_audio


def main():
    signal, sampling_rate = librosa.load("/home/b4nsh33/data_plot/acoustic_guitar/MEDLEYDB_AimeeNorwich_Child_RAW_07_01.wav")
    melspectrogram = mel_spectrogram_from_raw_audio(signal[0:10*sampling_rate], sampling_rate)
    plt.grid()

    ax1 = plt.subplot(231)
    librosa.display.specshow(librosa.logamplitude(melspectrogram),
                             y_axis='mel',
                             x_axis='time',
                             sr=sampling_rate,
                             hop_length=settings.HOP_LENGTH,
                             fmin=0,
                             fmax=10000)
    plt.title('Acoustic Guitar')
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.tight_layout()


    signal, sampling_rate = librosa.load(
        "/home/b4nsh33/data_plot/overhead/ENST_D1_L_036_phrase_disco_simple_slow_sticks.wav")
    melspectrogram = mel_spectrogram_from_raw_audio(signal[0:10 * sampling_rate], sampling_rate)
    ax2 = plt.subplot(232)
    librosa.display.specshow(librosa.logamplitude(melspectrogram),
                             y_axis='mel',
                             x_axis='time',
                             sr=sampling_rate,
                             hop_length=settings.HOP_LENGTH,
                             fmin=0,
                             fmax=10000)
    plt.title('Overhead')
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.tight_layout()


    signal, sampling_rate = librosa.load(
        "/home/b4nsh33/data_plot/vocals_male/MEDLEYDB_Auctioneer_OurFutureFaces_RAW_08_01.wav")
    melspectrogram = mel_spectrogram_from_raw_audio(signal[0:10 * sampling_rate], sampling_rate)
    ax3 = plt.subplot(233)
    librosa.display.specshow(librosa.logamplitude(melspectrogram),
                             y_axis='mel',
                             x_axis='time',
                             sr=sampling_rate,
                             hop_length=settings.HOP_LENGTH,
                             fmin=0,
                             fmax=10000)
    plt.title('Vocals Male')
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.tight_layout()


    signal, sampling_rate = librosa.load(
        "/home/b4nsh33/data_plot/electric_guitar/MEDLEYDB_Creepoid_OldTree_RAW_05_01.wav")
    melspectrogram = mel_spectrogram_from_raw_audio(signal[0:10 * sampling_rate], sampling_rate)
    ax4 = plt.subplot(234)
    librosa.display.specshow(librosa.logamplitude(melspectrogram),
                             y_axis='mel',
                             x_axis='time',
                             sr=sampling_rate,
                             hop_length=settings.HOP_LENGTH,
                             fmin=0,
                             fmax=10000)
    plt.title('Electric Guitar')
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.tight_layout()


    signal, sampling_rate = librosa.load(
        "/home/b4nsh33/data_plot/bass/MEDLEYDB_FamilyBand_Again_RAW_01_01.wav")
    melspectrogram = mel_spectrogram_from_raw_audio(signal[0:10 * sampling_rate], sampling_rate)
    ax5 = plt.subplot(235)
    librosa.display.specshow(librosa.logamplitude(melspectrogram),
                             y_axis='mel',
                             x_axis='time',
                             sr=sampling_rate,
                             hop_length=settings.HOP_LENGTH,
                             fmin=0,
                             fmax=10000)
    plt.title('Bass')
    plt.setp(ax5.get_xticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.tight_layout()



    signal, sampling_rate = librosa.load(
        "/home/b4nsh33/data_plot/vocals_female/MEDLEYDB_ClaraBerryAndWooldog_AirTraffic_RAW_07_01.wav")
    melspectrogram = mel_spectrogram_from_raw_audio(signal[0:10 * sampling_rate], sampling_rate)
    ax6 = plt.subplot(236)
    librosa.display.specshow(librosa.logamplitude(melspectrogram),
                             y_axis='mel',
                             x_axis='time',
                             sr=sampling_rate,
                             hop_length=settings.HOP_LENGTH,
                             fmin=0,
                             fmax=10000)
    plt.title('Vocals Female')
    plt.setp(ax6.get_xticklabels(), visible=False)
    plt.setp(ax6.get_yticklabels(), visible=False)
    plt.tight_layout()



    # plt.show(True)
    plt.savefig("/home/b4nsh33/instruments_spectrogram.png")

if __name__ == "__main__":
    main()
