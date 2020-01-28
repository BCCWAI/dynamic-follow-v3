import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import time
import copy
from tokenizer import tokenize
import seaborn as sns
# import load_brake_pred_model_corolla as brake_wrapper
from even_out_distribution import even_out_distribution

# brake_model, brake_scales = brake_wrapper.get_brake_pred_model()

def interp_fast(x, xp, fp):  # extrapolates above range, np.interp does not
    return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]


os.chdir("C:/Git/dynamic-follow-tf-v2/data")
data_folders = ["D:/Resilio Sync/df", "D:/Resilio Sync/dfv2"]
driving_data = []
supported_users = ['CHEVROLET VOLT']
consider_set_speed = False  # removing set_speed for now

print("Loading data...")
for data_dir in data_folders:
    old_data = 'dfv2' not in data_dir
    for folder in [i for i in os.listdir(data_dir) if any([x in i for x in supported_users])]:
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if 'old' not in filename and '.txt' not in filename and 'df-data' in filename:
                file_path = os.path.join(os.path.join(data_dir, folder), filename)
                if os.path.getsize(file_path) / 1e6 < 10:
                    continue
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
                    if not old_data:
                        keys = data[0]  # gets keys and removes keys from data
                        if len(keys) != len(data[1]):
                            print('Length of keys not equal to length of data')
                            raise Exception
                        if 'track_data' in keys:
                            keys[keys.index('track_data')] = 'live_tracks'
                        if 'status' in keys:
                            keys[keys.index('status')] = 'lead_status'
                        if 'time.time()' in keys:
                            keys[keys.index('time.time()')] = 'time'
                        data = data[1:]
                        data = [dict(zip(keys, i)) for i in data]
                    else:
                        if len(data[0]) == 9:
                            keys = ['v_ego', 'a_ego', 'v_lead', 'x_lead', 'a_lead', 'a_rel', 'gas', 'brake', 'time']
                        else:
                            print("This file's keys do not match the standard, skipping!")
                            print('Missed data: {} Mb'.format(round(os.path.getsize(file_path) / 1e6), 3))
                            continue
                        keys = ['v_ego', 'a_ego', 'v_lead', 'x_lead', 'a_lead', 'a_rel', 'gas', 'brake', 'time']
                        data = [dict(zip(keys, i)) for i in data]
                else:
                    raise Exception("Error. Not new format!")

                for line in data:  # map gas and brake values to appropriate 0 to 1 range and append to driving_data
                    if consider_set_speed and (line['set_speed'] == 0.0 or (line['set_speed'] > line['v_ego'] and line['car_gas'] > .15)):
                        continue

                    if 'HOLDEN' not in folder and not old_data:
                        line['gas'] = float(line['car_gas'])

                    if 'HOLDEN' not in folder and 'toyota' in folder.lower():
                        new_brake = line['brake'] / 4047.0 if line['brake'] >= 256 and line['gas'] == 0.0 else 0  # throw out brake when gas is applied or pressure less than or equal to 512
                        line.update({'brake': new_brake})
                    # line['v_ego'] = max(line['v_ego'], 0.0)  # remove negative velocities
                    driving_data.append(line)

predict_brake = False
if predict_brake:
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
    avg_time = sum(avg_times) / len(avg_times)
    print("Average time: {}".format(round(avg_time, 5)))

    seq_time = 0.5
    seq_len = round(seq_time / avg_time)

    data_sequences = []
    for seq in data_split:
        data_sequences += tokenize(seq, seq_len)

    print("Predicting brake samples...", flush=True)
    x_train = []
    y_train = []
    count = 0
    pos_preds = 0
    neg_preds = 0
    for idx, seq in enumerate(data_sequences):
        if count > len(data_sequences) / 10:
            print("{}% samples predicted!".format(round(idx/len(data_sequences), 2)))
            count = 0
        x_train.append(seq[0])
        if seq[0]['gas'] - seq[0]['brake'] < 0:  # only predict y_train if not coasting or accelerating
            to_pred = np.interp(np.array([sample['v_ego'] for sample in seq]), brake_scales['v_ego'], [0, 1])
            predicted_brake = np.interp(brake_model.predict([[to_pred]])[0][0], [0, 1], brake_scales['gas'])
            if predicted_brake <= 0:
                neg_preds += 1
            else:
                pos_preds += 1
                predicted_brake = -0.2
            y_train.append(predicted_brake)
        else:
            y_train.append(seq[0]['gas'])  # we can assume gas is activated, or if it's not, then we're coasting
        count += 1
    print('Of {} predictions, {} were incorrectly positive while {} were correctly negative.'.format(pos_preds + neg_preds,
                                                                                                     pos_preds, neg_preds))


y_train = [i['gas'] - i['brake'] for i in driving_data]

remove_keys = ['gas', 'brake', 'v_lat', 'car_gas', 'path_curvature', 'decel_for_model', 'live_tracks']  # remove these unneeded keys in training
x_train = [{key: line[key] for key in line if key not in remove_keys} for line in driving_data]  # remove gas/brake from x_train

remove_brake = False
if remove_brake:
    x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if y >= 0.0])
    x_train, y_train = map(list, [x_train, y_train])

# x_train, y_train = even_out_distribution(x_train, y_train, n_sections=15, m=2, reduction=.75, reduce_min=.4)

print("Total samples: {}".format(len(y_train)))
print("Gas samples: {}".format(len([i for i in y_train if i > 0])))
print("Coast samples: {}".format(len([i for i in y_train if i == 0])))
print("Brake samples: {}".format(len([i for i in y_train if i < 0])))

average_y = [i for i in y_train]
average_y = sum(average_y) / len(average_y)
print('Average of samples: {}'.format(average_y))
#plt.plot(range(len(driving_data)), [i['decel_for_model'] for i in driving_data])


save_data = True
if save_data:
    print("Saving data...")
    save_dir = "simple_model"
    # x_train = [{'v_ego': sample['v_ego'], 'v_lead': sample['v_lead'], 'a_ego': sample['a_ego'], 'x_lead': sample['x_lead'], 'a_lead': sample['a_lead']} for sample in driving_data]
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
