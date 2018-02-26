import os

from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

from ba_code import settings
from ba_code.utils import utils


class TestSetScorer:
    def __init__(self, X_test, y_test, test_metainfo, y_mapping, config, result_dir, file_prefix):
        self.X_test = X_test
        self.y_test = y_test
        self.metainfo = test_metainfo
        self.y_mapping = y_mapping
        self.save_plots = config.get('save_plots', False)

        file_name = file_prefix + '_nn_info.txt'
        self.path_to_saved_file = os.path.join(result_dir, file_name)

    # The call method will be called from nolearn when all epochs trained
    def __call__(self, nn, train_history):
        print "Testing net..."
        score = self.score(nn)
        print('Accuracy test score is %.4f' % score)
        if self.save_plots:
            # append info to file
            info_file = open(self.path_to_saved_file, 'a')
            output = "--------------------------------------\n"
            output += "Score on testset is: {0}\n".format(str(score))
            info_file.write(output)
            info_file.close()

    def score(self, net):
        Xb, yb, metainfo = utils.extract_mini_segments_with_metainfo(self.X_test, self.y_test, self.metainfo)
        Xb, yb, metainfo = utils.filter_mini_segments_with_metainfo(Xb, yb, metainfo)
        predictions = net.predict(Xb)
        file_output = "--------------------------------------\n"
        for i in range(len(predictions)):
            if predictions[i] != yb[i]:
                dt = datetime.now()
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0)
                dt = dt + timedelta(seconds=(metainfo[i][2] * settings.MINI_BATCH_SECONDS))
                output = "{0} classified as {1}; look at {2}:{3}".format(metainfo[i][0][28:],
                                                                         self.y_mapping[predictions[i]],
                                                                         dt.minute, dt.second)
                print output
                file_output += output + "\n"

        if self.save_plots:
            # append info to file
            info_file = open(self.path_to_saved_file, 'a')
            file_output += '\n'
            info_file.write(file_output)
            info_file.close()
        return float(accuracy_score(predictions, yb))
