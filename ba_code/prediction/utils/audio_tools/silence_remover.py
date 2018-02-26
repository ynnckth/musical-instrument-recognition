
# Use this script to remove silence parts from wav-uploads.
# Usage: silence_remover in.wav out.wav

from pydub import AudioSegment
from pydub.silence import split_on_silence
import sys


arguments = sys.argv
file_name = ""
out_file = ""

if len(arguments) < 3:
    exit("usage: silence_remover AUDIO-FILE")
elif len(arguments) == 3:
    file_name = arguments[1]
    out_file = arguments[2]


def remove_silence(sound, min_silence_time=1000, silence_treshold=-55):
    chunks = split_on_silence(sound,
                              # must be silent for at least half a second
                              min_silence_len=min_silence_time,

                              # consider it silent if quieter than silence_threshhold
                              silence_thresh=silence_treshold
                              )

    # Concatenate the chunks without silence
    without_silence = chunks[0]
    for i in range(1, len(chunks)):
        without_silence = without_silence + chunks[i]

    return without_silence


def main():
    audio_signal = AudioSegment.from_file(file_name, format="wav")

    print "Removing silence from file ", file_name, " ..."
    without_silence = remove_silence(audio_signal)

    print "Exporting new track as ", out_file, " ..."
    without_silence.export(out_file, format("wav"))

    print "done."


if __name__ == "__main__":
    main()
