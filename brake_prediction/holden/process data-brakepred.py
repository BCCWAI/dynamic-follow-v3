import json
import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import csv
import time

os.chdir("C:\Git\dynamic-follow-tf-v2\data")
'''with open("traffic-highway/df-data", "r") as f:
    d_data = f.read().split("\n")'''

data_dir = "D:\Resilio Sync\df"
driving_data = []
gm_counter = 0
other_counter = 0

CHEVY = False
REMOVE_COAST_CHEVY = False

HONDA = False
HOLDEN = True
MINSIZE = 40000 #kb #40000
print("Loading data...")
for folder in os.listdir(data_dir):
    if HOLDEN and any([sup_car in folder for sup_car in ["HOLDEN ASTRA RS-V BK 2017"]]):
        for filename in os.listdir(os.path.join(data_dir, folder)):
            print(filename)
            file_size = os.path.getsize(os.path.join(os.path.join(data_dir, folder), filename))
            if file_size > MINSIZE: #if bigger than 40kb
                #print(os.path.join(os.path.join(data_dir, folder), filename))
                with open(os.path.join(os.path.join(data_dir, folder), filename), "r") as f:
                    df = [i for i in f.read().split("\n") if i != '']
                
                #df = [i for i in df if -1 <= (i[-3] - i[-2]) <= 1]
                
                for line in df:
                    try:
                        line = json.loads(line)
                    except:
                        continue
                    if line[-3] - line[-2] < 0.0:
                        driving_data.append(line)
                        gm_counter += 1

x = range(len(driving_data))
#y = [i[-3] - i[-2] for i in driving_data]
y2 = [i[0] for i in driving_data]
plt.clf()
#plt.plot(x, y)
plt.plot(x, y2)
plt.show()

print("Total samples: {}".format(len(driving_data)))
y_train = [i[-3] - i[-2] for i in driving_data] # since some samples have a_rel, get gas and brake from end of list
print("Gas samples: {}".format(len([i for i in y_train if i > 0])))
print("Coast samples: {}".format(len([i for i in y_train if i == 0])))
print("Brake samples: {}".format(len([i for i in y_train if i < 0])))
print("\nSamples from GM: {}, samples from other cars: {}".format(gm_counter, other_counter))

average_y = [i for i in y_train]
average_y = sum(average_y) / len(average_y)
print('Average of samples: {}'.format(average_y))

save_data = True
if save_data:
    print("Saving data...")
    save_dir="brake_pred"
    x_train = [[i[0], i[1]] for i in driving_data] # include a_rel
    #x_train = [i[:2] + [i[2] - i[0]] + i[-2:] for i in x_train] # makes index 2 be relative velocity
    with open(save_dir+"/x_train", "wb") as f:
        pickle.dump(np.array(x_train), f)
    with open(save_dir+"/y_train", "wb") as f:
        pickle.dump(np.array(y_train), f)
    print("Saved data!")

'''driving_data = [i for idx, i in enumerate(driving_data) if 20000 < idx < 29000]
x = [i for i in range(len(driving_data))]
y = [i[0] for i in driving_data]
plt.plot(x, y)
plt.show()'''