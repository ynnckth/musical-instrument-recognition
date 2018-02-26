
# Use this script to split a large wav file in multiple
# smaller (10 min) uploads.
# Usage: wav_file_splitter in.wav out_path

from pydub import AudioSegment
import os.path
import sys

arguments = sys.argv
file_name = ""

if len(arguments) != 3:
    exit("Usage: wav_file_splitter wav_file out_file")
elif len(arguments) == 3:
    file_name = arguments[1]
    out_path = arguments[2]


def main():
    print "importing wav file ", file_name, " ..."
    wav_file = AudioSegment.from_file(file_name, format="wav")
    wav_file_duration_ms = wav_file.duration_seconds * 1000

    ten_minutes_in_ms = 10 * 60 * 1000

    num_of_chunks = wav_file_duration_ms / ten_minutes_in_ms
    print "number of chunks to split: ", num_of_chunks

    path, basename = os.path.split(file_name)

    for i in range(0, int(num_of_chunks) + 1):
        print "splitting chunk nr. ", i, " ..."
        if i == num_of_chunks:
            chunk = wav_file[i*ten_minutes_in_ms:]
        else:
            chunk = wav_file[i*ten_minutes_in_ms:(i+1)*ten_minutes_in_ms]
        chunk.export(out_path + str(i) + "_" + basename, format("wav"))

    print "done."


if __name__ == "__main__":
    main()




