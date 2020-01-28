import numpy as np
import time

class FormatData:
    def __init__(self, model_inputs, samples_in_future):
        self.model_inputs = model_inputs
        self.samples_in_future = samples_in_future

    def format(self, driving_sequences):
        x_train = [self.get_values_from_keys(seq) for seq in driving_sequences]  # slightly quicker
        # x_train = list(map(self.get_values_from_keys, driving_sequences))

        y_train = [seq[-1]['gas'] - seq[-1]['brake'] for seq in driving_sequences]

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train

    def get_values_from_keys(self, seq):
        if self.samples_in_future != 0:
            to_return = [[sample[key] for key in self.model_inputs] for sample in seq][:-self.samples_in_future]  # removes unneeded keys and samples from end
        else:
            to_return = [[sample[key] for key in self.model_inputs] for sample in seq]
        return to_return


# fd = FormatData(['a_ego'], 0)
#
# fd.format([[{'a_ego': 5}, {'a_ego': 5}, {'a_ego': 5}, {'a_ego': 5}, {'a_ego': 5}, {'a_ego': 5}, {'a_ego': 5},
#                    {'a_ego': 5}, {'a_ego': 5}, {'a_ego': 5}, {'a_ego': 5}, {'a_ego': 5},
#                    {'a_ego': -4, 'gas': 3, 'brake': 2}] * 700000])
