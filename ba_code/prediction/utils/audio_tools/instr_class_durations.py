import csv
import os
import wave


DATA_ROOT_DIR = "/home/nutella/BA/data/"


def get_instrument_file_paths(root_dir):
    instrument_file_paths = dict()
    for entry in os.listdir(root_dir):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path):
            instrument_file_paths[entry] = []
            for instrument_file in os.listdir(entry_path):
                if instrument_file.endswith(".wav"):
                    file_path = os.path.join(entry_path, instrument_file)
                    instrument_file_paths[entry].append(file_path)

    return instrument_file_paths


# calculates the duration of each instrument class in the dataset in seconds
def calc_durations(instrument_files):
    print "calculating duration..."
    instr_class_durations = dict()
    for instrument_class, wav_file_paths in instrument_files.items():
        instr_class_durations[instrument_class] = 0
        for wav_file_path in wav_file_paths:
            wav_file = wave.open(wav_file_path, 'r')
            wav_duration = wav_file.getnframes() / float(wav_file.getframerate())   # in seconds
            instr_class_durations[instrument_class] += wav_duration

    print "done."
    return instr_class_durations


def export_as_csv(durations):
    durations_csv = open('durations.csv', 'wb')
    csv_writer = csv.DictWriter(durations_csv, durations.keys())
    csv_writer.writeheader()
    csv_writer.writerow(durations)
    durations_csv.close()


def main():
    instrument_files = get_instrument_file_paths(DATA_ROOT_DIR)
    instr_class_durations = calc_durations(instrument_files)
    export_as_csv(instr_class_durations)


if __name__ == "__main__":
    main()
