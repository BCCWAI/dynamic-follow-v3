import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tokenizer import tokenize
import pickle
import sys
import time

os.chdir("C:/Git/dynamic-follow-tf/data")
'''with open("traffic-highway/df-data", "r") as f:
    d_data = f.read().split("\n")'''

data_dir = "D:\Resilio Sync\df"
d_data = []
gm_counter = 0
other_counter = 0

CHEVY = True
REMOVE_COAST_CHEVY = False

HONDA = False
HOLDEN = False
MINSIZE = 40000 #kb
for folder in os.listdir(data_dir):
    if any([sup_car in folder for sup_car in ["CHEVROLET VOLT PREMIER 2017"]]) and CHEVY:
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > MINSIZE: #if bigger than 40kb
                #print(os.path.join(os.path.join(data_dir, folder), filename))
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                
                started = False
                for line in df:
                    try:
                        line = json.loads(line)
                    except:
                        continue
                    line[0] = max(round(line[0], 12), 0.0)
                    if not started and line[0] != 0.0:
                        started = True
                    if started:
                        if REMOVE_COAST_CHEVY and line[-2] + line[-3] == 0.0: # skip coasting samples since has regen
                            continue
                        line[1] = round(line[1], 12) #a_ego
                        line[2] = max(round(line[2], 12), 0.0)
                        line[3] = max(round(line[3], 12), 0.0)
                        line[4] = round(line[4], 12) #a_lead
                        gm_counter+=1
                        d_data.append(line)
    
    elif any([sup_car in folder for sup_car in ["HOLDEN ASTRA RS-V BK 2017"]]) and HOLDEN:
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > MINSIZE: #if bigger than 40kb
                #print(os.path.join(os.path.join(data_dir, folder), filename))
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                
                started = False
                for line in df:
                    try:
                        line = json.loads(line)
                    except:
                        continue
                    line[0] = max(round(line[0], 12), 0.0)
                    if not started and line[0] != 0.0:
                        started = True
                    if started:
                        line[1] = round(line[1], 12) #a_ego
                        line[2] = max(round(line[2], 12), 0.0)
                        line[3] = max(round(line[3], 12), 0.0)
                        line[4] = round(line[4], 12) #a_lead
                        gm_counter+=1
                        d_data.append(line)
    
    elif any([sup_car in folder for sup_car in ["HONDA CIVIC 2016 TOURING"]]) and HONDA:
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > MINSIZE: #if bigger than 40kb
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                
                started = False
                for line in df:
                    try:
                        line = json.loads(line)
                    except:
                        continue
                    line[0] = max(round(line[0], 12), 0.0)
                    if not started and line[0] != 0.0:
                        started = True
                    if started:
                        line[1] = round(line[1], 12) #a_ego
                        line[2] = max(round(line[2], 12), 0.0)
                        line[3] = max(round(line[3], 12), 0.0)
                        line[4] = round(line[4], 12) #a_lead
                        if line[-2] > 1.0 and line[0] < .01: # user brake skyrockets when car is stopped for some reason, though since we have data from gm, we can exclude this data from honda
                            pass
                        else:
                            line[-2] = np.clip(line[-2], 0.0, 1.0) # sometimes goes neg when really no brake
                            if -1 <= (line[-3] - line[-2]) <= 1: # make sure gas/brake is in range
                                d_data.append(line)
                                other_counter+=1
                        
    
    # the following should improve performance for deciding when and how much to apply gas (but might reduce braking performance)
    '''elif any([sup_car in folder for sup_car in ["TOYOTA COROLLA 2017", "TOYOTA PRIUS 2017", "TOYOTA RAV4 HYBRID 2017", "TOYOTA RAV4 2017"]]):
        for filename in os.listdir(os.path.join(data_dir, folder)):
            if os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename)) > 40000: #if bigger than 40kb
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = f.read().split("\n")
                for line in df:
                    if line != "" and "[" in line and "]" in line and len(line) >= 40:
                        line = ast.literal_eval(line)
                        line[6] = 0.0  # don't include brake pressure
                        d_data.append(line)
                        other_counter+=1
                        #if line[6] == 0.0 or line[5] > 0.0:  # for cars without brake sensor (like toyotas), only include lines with no brake. brake pressure is too inaccurate
                            #line[6] = 0.0  # don't include brake pressure
                            #other_counter+=1
                            #d_data.append(line)  # need to experiment with including braking samples, but setting brake to 0 so the model will coast instead of not knowing what to do'''

driving_data = []
for line in d_data:  # do filtering
    if line[0] < -0.22352 or sum(line) == 0: #or (sum(line[:3]) == 0):
        continue
    if line[4] > 10 or line[4] < -10: # filter out crazy lead acceleration
        continue
    if line[-3] - line[-2] > .7 or line[-3] - line[-2] < -0.85:
        continue
    '''if line[-3] - line[-2] == 1.0 or line[-3] - line[-2] == -1.0:  # temp for now! (for honda)
        continue'''
    #line[0] = max(line[0], 0)
    #line[2] = max(line[2], 0)
    #line[3] = max(line[3], 0)
    
    #line[-1] = line[-1] / 4047.0  # only for corolla
    #line = [line[0], line[1], (line[2]-line[0]), line[3], line[4], line[5], line[6], line[7]] # this makes v_lead, v_rel instead
    driving_data.append(line)
print(len(driving_data))
#driving_data = driving_data[:5000000]

to_tokenize = True
if to_tokenize:
    print("Tokenizing...")
    sys.stdout.flush()
    seq_length = 20
    lead_split = [[]]
    counter = 0
    for idx, i in enumerate(driving_data):
        if idx != 0:
            if abs(i[3] - driving_data[idx-1][3]) > 2.5: # if new lead, split
                counter += 1
                lead_split.append([])
        lead_split[counter].append([i[0], i[2], i[3]])
    
    lead_split = [i for i in lead_split if len(i) >= seq_length] # remove sequences with len below seq_length
    lead_tokenized = [tokenize(i, seq_length) if len(i) > seq_length else [i] for i in lead_split] # tokenize data, ignore sequences if 10 already
    x_train = [inner for outer in lead_tokenized for inner in outer] # join nested sequences to one list
    
    if len([i for i in x_train if len(i) != seq_length]) != 0:
        print("Something is wrong with the tokenization...")
    print("After tokenizing, we have {} samples.".format(len(x_train)))
    #x_train = [[[x[0], x[2], x[3]] for x in i] for i in lead_tokenized] # keep only v_ego, v_lead, and x_lead
    y_train = [(i[-1][-3] - i[-1][-2]) for i in x_train] # last gas/brake val in sequence
    print('Got training variables', flush=True)
    even_out_gas = False
    if even_out_gas:  # makes number of gas/brake/nothing samples equal to min num of samples
        gas = [idx for idx, i in enumerate(y_train) if i > 0.0]
        coast = [idx for idx, i in enumerate(y_train) if i == 0.0]
        brake = [idx for idx, i in enumerate(y_train) if i < 0.0]
        
        to_remove_gas = len(gas) - min(len(gas), len(coast), len(brake)) if len(gas) != min(len(gas), len(coast), len(brake)) else 0
        to_remove_coast = len(coast) - min(len(gas), len(coast), len(brake)) if len(coast) != min(len(gas), len(coast), len(brake)) else 0
        to_remove_brake = len(brake) - min(len(gas), len(coast), len(brake)) if len(brake) != min(len(gas), len(coast), len(brake)) else 0
        
        del gas[:to_remove_gas]
        del coast[:to_remove_coast]
        del brake[:to_remove_brake]
        
        indexes = gas + coast + brake
        x_train = [x_train[i] for i in indexes]
        y_train = [y_train[i] for i in indexes] # gets shuffled before training, so we don't have to worry about shuffling
    
    #print("Gas samples: {}".format(len([idx for idx, i in enumerate(y_train) if i > 0.0])))
    #print("Coast samples: {}".format(len([idx for idx, i in enumerate(y_train) if i == 0.0])))
    #print("Brake samples: {}".format(len([idx for idx, i in enumerate(y_train) if i < 0.0])))
    sys.stdout.flush()
    save_data = True
    if save_data:
        print('Writing data', flush=True)
        np.save('LSTM/x_train', x_train)
        np.save('LSTM/y_train', y_train)
        print("Saved data!")

verbose = True
if verbose:
    print(len(driving_data))
    print()
    #y_train = [i[5] - i[6] for i in driving_data]
    y_train = [i[-3] - i[-2] for i in driving_data] # since some samples have a_rel, get gas and brake from end of list
    print("Gas samples: {}".format(len([i for i in y_train if i > 0])))
    print("Coast samples: {}".format(len([i for i in y_train if i == 0])))
    print("Brake samples: {}".format(len([i for i in y_train if i < 0])))
    print("\nSamples from GM: {}, samples from other cars: {}".format(gm_counter, other_counter))