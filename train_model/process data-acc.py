import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import time
from sys import getsizeof
import gc

def interp_fast(x, xp, fp):  # extrapolates above range, np.interp does not
    return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]

os.chdir("C:/Git/dynamic-follow-tf-v2/data")
data_dir = "D:/Resilio Sync/dfv2"
driving_data = []
supported_users = ['ShaneSmiskol-TOYOTA COROLLA 2017']
consider_set_speed = False  # removing set_speed for now
s = time.time()
print("Loading data...")
for folder in [i for i in os.listdir(data_dir) if i in supported_users]:
    for idx, filename in enumerate(os.listdir(os.path.join(data_dir, folder))):
        if 'old' not in filename and '.txt' not in filename:
            driving_data.append([])
            print(idx)
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
                if 'track_data' in keys:
                    keys[keys.index('track_data')] = 'live_tracks'
                if 'lead_status' in keys:
                    keys[keys.index('lead_status')] = 'status'
                data = data[1:]
                data = [dict(zip(keys, i)) for i in data]

            # pedal = any([True if i['car_gas'] > 0.0 else False for i in data])
            # if pedal:
            pedal_gas = [line['gas'] for line in data]  # data from comma pedal
            car_gas = [line['car_gas'] for line in data]  # data from car's can

            max_car_gas = max(car_gas)  # this is most accurate way to map pedal gas to car gas range (0 to 1)
            max_pedal_gas = pedal_gas[car_gas.index(max_car_gas)]  # comma pedal has more resolution than car's sensor

            min_car_gas = [i for i in car_gas if i > 0.0][0]
            min_pedal_gas = pedal_gas[car_gas.index(min_car_gas)]

            pedal_gas = [interp_fast(i, [min_pedal_gas, max_pedal_gas], [min_car_gas, max_car_gas]) for i in pedal_gas]
            pedal_gas = [i if i >= 0 else 0 for i in pedal_gas]
            # else:
                # print('Not pedal!')

            for line in data:  # map gas and brake values to appropriate 0 to 1 range and append to driving_data
                if (line['set_speed'] == 0.0 or (line['set_speed'] > line['v_ego'] and line['car_gas'] > .15)) and consider_set_speed:
                    continue
                # if pedal:
                new_gas = (interp_fast(line['gas'], [min_pedal_gas, max_pedal_gas], [min_car_gas, max_car_gas]) * 0.4) + (line['car_gas'] * 0.6)  # makes it so that it's closer to the car's gas sensor while still preserving some extra accuracy (noise?) from pedal sensor
                new_gas = new_gas if new_gas >= 0 and line['car_gas'] != 0 else 0  # if remapped gas is negative, no gas
                line.update({'gas': new_gas})
                #

                new_brake = line['brake'] / 4047.0 if line['brake'] >= 256 and line['gas'] == 0.0 else 0  # throw out brake when gas is applied or pressure less than or equal to 512
                line.update({'brake': new_brake})
                driving_data[idx].append(line)

#print(time.time()-s)

driving_data_new = []
offset_by = 5  # about what a_ego is offset by, in samples
for seq in driving_data:
    seq_builder = []
    for idx, sample in enumerate(seq):
        if idx + offset_by >= len(seq):
            break
        sample = dict(sample)
        sample['a_ego'] = seq[idx + offset_by]['a_ego']
        seq_builder.append(sample)
    driving_data_new.append(seq_builder)

driving_data = [item for sublist in driving_data_new for item in sublist]

# plt.plot(range(len(driving_data[0])), [i['a_ego'] for i in driving_data[0]], label='old')
# plt.plot(range(len(driving_data_new[0])), [i['a_ego'] for i in driving_data_new[0]], label='new')
# plt.legend()
# plt.show()
# print("HERE")
# print(driving_data_new[0][3546]['a_ego'], driving_data[0][3536]['a_ego'])
# I've identified the average spread of time differences to be within a range below,
# so filter out large diffs where the file was being written, or other unknown factors occurred to deviate the difference

'''r_t = [0.03, 0.07]  # rate tolerance, include times ≈ 20Hz, avg. rate than long_mpc runs at
target_rate = 0.05
rate_tolerance = 0.02

seq = [1, 2, 3, 4, 5, 6, 7, 90, 91, 92, 93, 94, 97, 98, 150, 151, 153]
a = [[]]
cur_idx = 0
for idx, sample in enumerate(driving_data[0]):
    if sample['time'] - seq[idx - 1]['time'] > r_t[0]:
        a.append([])
        cur_idx += 1
    if sample['time'] - seq[idx - 1]['time'] < r_t[1]:
        a[cur_idx].append(sample)


# For each sequence (drive), we iterate through samples and calculate the difference of time between the current sample and previous sample
driving_data_filtered = []
for seq_idx, seq in enumerate(driving_data):
    filtered = [sample for idx, sample in enumerate(seq) if r_t[0] <= (sample['time'] - seq[idx - 1]) <= r_t[1]]
    driving_data_filtered.append(filtered)
    #diffs = [{'time_difference': sample['time'] - seq[idx - 1]['time'], 'sample': sample} for idx, sample in enumerate(seq) if idx > 0]
    #driving_data_filtered.append([i for i in diffs if target_rate-rate_tolerance <= i['time_difference'] <= target_rate+rate_tolerance])

# Now split into leads

#  Now calculate accurate acceleration
samples_to_calc = 1
driving_data_acc = [[]] * len(driving_data_filtered)
for seq_idx, seq in enumerate(driving_data_filtered):
    for idx, sample in enumerate(seq):
        if idx > 0:
            time_difference = sample['time_difference']
            sample = sample['sample']
            sample.update({'new_acc': (sample['v_ego'] - seq[idx - samples_to_calc]['sample']['v_ego']) / (sample['time'] - seq[idx - samples_to_calc]['sample']['time'])})
            driving_data_acc[seq_idx].append(sample)
        #[(sample['v_ego'] - seq[idx - samples_to_calc]['v_ego']) / (sample['time'] - seq[idx - samples_to_calc]['time']) for idx, sample in enumerate(seq) if idx > 0]
    #driving_data_acc.append(seq)

plt.clf()
plt.plot(range(len(driving_data_acc[2])), [i['a_ego'] for i in driving_data_acc[2]], label='before')
plt.plot(range(len(driving_data_acc[2])), [i['new_acc'] for i in driving_data_acc[2]], label='after')
plt.legend()
plt.show()


# times = [i['time'] for idx, i in enumerate(driving_data[seq_idx]) if idx > 0]
# plt.clf()
# plt.plot(range(len(diffs)), [i[1] for i in diffs], label='after')
# plt.plot(range(len(times)), times, label='before')
# plt.legend()
# plt.show()

# for i in range(len(driving_data)):
#     plt.clf()
#     plt.plot(range(len(driving_data[i])), [i['time'] for i in driving_data[i]])
#     plt.show()
#     plt.pause(.01)
#     input()'''

'''#  Split data into sections based on time difference
driving_data_split = [[]]
counter = 0
for idx, i in enumerate(driving_data):
    if idx != 0:
        if .04 > i['time'] - driving_data[idx-1]['time'] > .06:  # new lead, split
            counter += 1
            driving_data_split.append([])
    driving_data_split[counter].append(i)

#  Calculate new a_ego from previous speed
new_driving_data = []
samples_to_calc = 1
for sequence in driving_data_split:
    for idx, sample in enumerate(sequence):
        if idx >= samples_to_calc:
            a_ego_new = (sample['v_ego'] - sequence[idx-samples_to_calc]['v_ego']) / (sample['time'] - sequence[idx-samples_to_calc]['time'])
            if a_ego_new > 5:
                print(sample)
                print()
                print(sequence[idx-1])
                print(a_ego_new)
                print('\n')
                break
            sample.update({'a_ego_new': a_ego_new})  # add new a_ego
            new_driving_data.append(sample)
plt.clf()
plt.plot(range(len(new_driving_data)), [i['a_ego'] for i in new_driving_data], label='old a_ego')
plt.plot(range(len(new_driving_data)), [i['a_ego_new'] for i in new_driving_data], label='new a_ego')
plt.plot(range(len(new_driving_data)), [i['v_ego']/15 for i in new_driving_data], label='velocity')
plt.plot(range(len(new_driving_data)), [0]*len(new_driving_data))
plt.legend()
plt.show()


'''
#plt.plot(range(len(driving_data)), [i['decel_for_model'] for i in driving_data])

# new_driving_data = []
# samples_to_calc = 2
# for idx, i in enumerate(driving_data):
#     if idx >= samples_to_calc:
#         if (i['time'] - driving_data[idx - samples_to_calc]['time']) < 0.09:  # average time is 0.0503 seconds, so .1 should work well
#             new_a_ego = (i['v_ego'] - driving_data[idx - samples_to_calc]['v_ego']) / (i['time'] - driving_data[idx - samples_to_calc]['time'])
#             if i['time'] - driving_data[idx - samples_to_calc]['time'] < 0:
#                 print(i['time'] - driving_data[idx - samples_to_calc]['time'])
#             i.update({'a_ego_car': i['a_ego']})
#             i.update({'a_ego': new_a_ego})
#             new_driving_data.append(i)
# 
# a_ego_old = [i['a_ego_car'] for i in new_driving_data]
# a_ego_new = [i['a_ego'] for i in new_driving_data]
# 
# plt.clf()
# plt.plot(range(len(a_ego_new)), a_ego_old, label='a_ego')
# plt.plot(range(len(a_ego_new)), a_ego_new, label='new acc')
# plt.legend()
# plt.show()

print("Total samples: {}".format(len(driving_data)))
y_train = [line['gas'] - line['brake'] for line in driving_data]
print("Gas samples: {}".format(len([i for i in y_train if i > 0])))
print("Coast samples: {}".format(len([i for i in y_train if i == 0])))
print("Brake samples: {}".format(len([i for i in y_train if i < 0])))

average_y = [i for i in y_train]
average_y = sum(average_y) / len(average_y)
print('Average of samples: {}'.format(average_y))

save_data = True
if save_data:
    print("Saving data...")
    save_dir = "live_tracks"
    x_train = [{key: line[key] for key in line if key not in ['gas', 'brake']} for line in driving_data]  # remove gas/brake from x_train
    with open(save_dir+"/x_train", "wb") as f:
        pickle.dump(x_train, f)
    with open(save_dir+"/y_train", "wb") as f:
        pickle.dump(y_train, f)

    to_remove = ["/x_train_normalized", "/y_train_normalized"]
    for sample in to_remove:
        try:
            os.remove(save_dir + sample)
        except:
            pass
    print("Saved data!")