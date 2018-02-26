import numpy as np
from nolearn.lasagne import TrainSplit

from ba_code import settings
from ba_code.settings import VAL_TRAIN_RATIO
from ba_code.utils import utils


class InstrumentTrainSplit(TrainSplit):
    def __call__(self, X, y, net):

        # validation set is 20% of the whole data set
        val_ratio = VAL_TRAIN_RATIO
        val_size = int(val_ratio * len(X) / (1 - val_ratio))
        train_size = int(len(X) - val_size)

        num_instr = utils.get_num_of_instr(y)
        segments_per_instr_in_X = self.get_segments_per_instr(y)

        i_train = train_size / num_instr
        i_val = val_size / num_instr

        X_train = np.zeros((i_train * num_instr, 1, settings.SEGMENT_LENGTH, settings.MEL_DATA_POINTS), dtype=np.float32)
        y_train = np.zeros(i_train * num_instr, dtype=np.int32)

        X_val = np.zeros((i_val * num_instr, 1, settings.SEGMENT_LENGTH, settings.MEL_DATA_POINTS), dtype=np.float32)
        y_val = np.zeros(i_val * num_instr, dtype=np.int32)

        for i in range(0, num_instr):
            start_idx = i * segments_per_instr_in_X
            end_idx = start_idx + i_train

            X_train[i*i_train : i*i_train+i_train] = X[start_idx:end_idx]
            y_train[i*i_train : i*i_train+i_train] = y[start_idx:end_idx]
            X_val[i*i_val : i*i_val+i_val] = X[end_idx:end_idx + i_val]
            y_val[i*i_val : i*i_val+i_val] = y[end_idx:end_idx + i_val]

        return X_train, X_val, y_train, y_val

    @staticmethod
    def get_segments_per_instr(y):
        segments_per_instr = 0
        curr_instr = y[0]
        for i in range(0, len(y)):
            if curr_instr == y[i]:
                segments_per_instr += 1
            else:
                break
        return segments_per_instr
