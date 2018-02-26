import os

from ba_code.utils import utils


class ModelSaver:
    def __init__(self, config, result_dir, file_prefix, y_mapping):
        self.save_model = config.get('save_model', False)
        self.result_dir = result_dir
        self.file_prefix = file_prefix
        self.y_mapping = y_mapping

    # The call method will be called from nolearn when all epochs trained
    def __call__(self, nn, train_history):
        if self.save_model:
            model_fname = os.path.join(self.result_dir, self.file_prefix + "_net.pickle")
            weights_fname = os.path.join(self.result_dir, self.file_prefix + "_net_weigths.pickle")
            y_mapping_fname = os.path.join(self.result_dir, self.file_prefix + "_y_mapping.pickle")

            utils.save_as_pickle((nn, self.y_mapping), model_fname)
            print('Model saved to %s' % model_fname)

            utils.save_as_pickle(self.y_mapping, y_mapping_fname)
            print('Model y mapping saved to %s' % y_mapping_fname)

            nn.save_params_to(weights_fname)
            print('Model weigths saved to %s' % weights_fname)
        else:
            print('Model not saved')
