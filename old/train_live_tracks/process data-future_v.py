import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import time
from tokenizer import tokenize

def interp_fast(x, xp, fp):  # extrapolates above range, np.interp does not
    return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]


os.chdir("C:/Git/dynamic-follow-tf-v2/data")
data_dir = "D:/Resilio Sync/dfv2"
driving_data = []
supported_users = ['all']  # ['ShaneSmiskol-TOYOTA COROLLA 2017']  # , 'i9NmzGB44XW8h86-TOYOTA COROLLA 2017']  #,]
consider_set_speed = False  # removing set_speed for now
use_pedal = False

print("Loading data...")
for folder in [i for i in os.listdir(data_dir) if i not in supported_users]:
    for filename in os.listdir(os.path.join(data_dir, folder)):
        if 'df-data' in filename:
            file_path = os.path.join(os.path.join(data_dir, folder), filename)

            print('Processing: {}'.format(file_path))
            with open(file_path, 'r') as f:
                df_data = f.read().replace("'", '"').replace('False', 'false').replace('True', 'true')

            data = []
            for sample in df_data.split('\n'):
                try:
                    data.append(json.loads(sample))
                except:
                    pass

            new_format = type(data[0]) == list  # new space saving format, this will convert it to list of dicts
            if new_format:
                keys = data[0]  # gets keys and removes keys from data
                if len(keys) != len(data[1]):
                    raise Exception('Length of keys not equal to length of data!')
                if 'track_data' in keys:
                    keys[keys.index('track_data')] = 'live_tracks'
                if 'status' in keys:
                    keys[keys.index('status')] = 'lead_status'
                if 'time.time()' in keys:
                    keys[keys.index('time.time()')] = 'time'
                data = data[1:]
                data = [dict(zip(keys, i)) for i in data]
            else:
                raise Exception("Error. Not new format!")

            for line in data:  # map gas and brake values to appropriate 0 to 1 range and append to driving_data
                if 'HOLDEN' not in folder:
                    line['gas'] = float(line['car_gas'])

                line['v_ego'] = max(line['v_ego'], 0.0)  # remove negative velocities
                line['v_lead'] = max(line['v_lead'], 0.0)  # remove negative velocities
                driving_data.append(line)

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
            avg_times.append(x['time'] - i[idx - 1]['time'])
avg_time = round(sum(avg_times) / len(avg_times), 2)
print("Average time: {}".format(avg_time))

x_train, y_train = [], []

future_time = 1  # in seconds to use as y_train
future_time_samples = round(future_time / avg_time)
for seq in data_split:
    for i in range(future_time_samples):
        tokenized = tokenize(seq[i:][::future_time_samples], 2)  # 2 because this isn't a sequence model
        for x in tokenized:
            x_train.append(x[0])  # this is current sample
            y_train.append(x[1]['v_ego'] - x[0]['v_ego'])  # this is the velocity future_time_samples in the future

print("Total samples: {}".format(len(driving_data)))

average_y = [i for i in y_train]
average_y = sum(average_y) / len(average_y)

remove_keys = ['gas', 'brake', 'car_gas', 'path_curvature', 'decel_for_model']  # remove these unneeded keys in training
save_data = True
if save_data:
    print("Saving data...")
    save_dir = "live_tracks"
    x_train = [{key: sample[key] for key in sample if key not in remove_keys} for sample in x_train]  # remove gas/brake from x_train
    with open(save_dir+"/x_train", "wb") as f:
        pickle.dump(x_train, f)
    with open(save_dir+"/y_train", "wb") as f:
        pickle.dump(y_train, f)
    
    to_remove = ["/x_train_normalized", "/y_train_normalized"]
    for i in to_remove:
        try:
            os.remove(save_dir + i)
        except:
            pass
    print("Saved data!")