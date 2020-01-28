import json
import ast
import os
from tokenizer import tokenize
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import time

os.chdir("C:\Git\dynamic-follow-tf-v2\data")
'''with open("traffic-highway/df-data", "r") as f:
    d_data = f.read().split("\n")'''

data_dir = "D:\Resilio Sync\dfv2"
driving_data = []

print("Loading data...")
for folder in os.listdir(data_dir):
    if any([sup_car in folder for sup_car in ["ShaneSmiskol-TOYOTA COROLLA 2017"]]):
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if 'brake-data' in filename:
                print(filename)
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    data = f.read().replace("'", '"').split("\n")

                data.remove('')
                data_parsed = json.loads('[{}]'.format(', '.join(data)))

                keys = data_parsed[0]
                data = data_parsed[1:]
                driving_data += [dict(zip(keys, i)) for i in data]

# raise Exception
data_split = [[]]
counter = 0
for idx, line in enumerate(driving_data):
    if idx > 0:
        time_diff = line['time'] - driving_data[idx - 1]['time']
        if abs(time_diff) > 0.1:
            counter += 1
            data_split.append([])
    data_split[counter].append(line)

avg_times = []
for i in data_split:
    for idx, x in enumerate(i):
        if idx > 0:
            avg_times.append(x['time'] - i[idx-1]['time'])
# avg_time = sum(avg_times) / len(avg_times)
# print("Average time: {}".format(round(avg_time, 5)))
avg_time = 0.01  # openpilot runs longcontrol at 100hz, so this makes sense

seq_time = 0.5
seq_len = round(seq_time / avg_time)
desired_seq_len = round(seq_time / 0.05)  # in 20 hertz form, what dftf data is gathered at
data_sequences = []
h100_to_20 = int(.05 / .01)  # 5.0
for seq in data_split:
    if len(seq) >= seq_len:
        for i in range(h100_to_20):  # lets us use all data, rather than throwing out 4/5 of it. like another layer of tokenization
            # don't have to check seq[i:] is at least desired_seq_len, as tokenization return [] if it's too short, and nothing will be added
            data_sequences += tokenize(seq[i:][::h100_to_20], desired_seq_len)  # converts to 20 hz, only use data every 5 samples

x_train = np.array([[max(sample['v_ego'], 0) for sample in seq] for seq in data_sequences])
y_train = np.array([seq[0]['gas'] - seq[0]['brake'] for seq in data_sequences])  # todo: try getting first gas value instead of last. gas in past affects future, not other way around!

print("Total samples: {}".format(len(driving_data)))

average_y = [i for i in y_train]
average_y = sum(average_y) / len(average_y)
print('Average of samples: {}'.format(average_y))

save_data = True
if save_data:
    print("Saving data...")
    save_dir = "brake_pred-Corolla/{}"
    with open(save_dir.format("x_train"), "wb") as f:
        pickle.dump(np.array(x_train), f)
    with open(save_dir.format("y_train"), "wb") as f:
        pickle.dump(np.array(y_train), f)
    print("Saved data!")
