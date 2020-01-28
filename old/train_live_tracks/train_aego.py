'''from numpy.random import seed
seed(255)
from tensorflow import set_random_seed
set_random_seed(255)'''
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten
import numpy as np
import random
from normalizer import normX
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
from sklearn.model_selection import train_test_split
import shutil
import functools
import operator
from keras.models import load_model
import os
#import load_brake_pred_model as lbpm


def interp_fast(x, xp, fp=[0, 1], ext=False):  # extrapolates above range when ext is True
    interped = (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]
    return interped if ext else min(max(min(fp), interped), max(fp))


#brake_model, brake_scales = lbpm.get_brake_pred_model()


def pad_tracks(tracks, max_tracks):
    to_add = max_tracks - len(tracks)
    to_add_left = to_add - (to_add // 2)
    to_add_right = to_add - to_add_left
    to_pad = [[0, 0, 0]]
    #return tracks + (to_add * to_pad)
    return (to_pad * to_add_left) + tracks + (to_pad * to_add_right)


os.chdir("C:/Git/dynamic-follow-tf-v2")
data_dir = "live_tracks"
norm_dir = "data/{}/normalized"
model_name = "live_tracks"

try:
    shutil.rmtree("models/h5_models/{}".format(model_name))
except:
    pass

'''for idx, i in enumerate(x_train):
    if y_train[idx] < 0.0:
        y_train[idx] = -0.2'''


if os.path.exists("data/{}/x_train_normalized".format(data_dir)):
    print('Loading normalized data', flush=True)
    with open("data/{}/x_train_normalized".format(data_dir), "rb") as f:
        tracks_normalized, car_data_normalized, scales = pickle.load(f)
    with open("data/{}/y_train_normalized".format(data_dir), "rb") as f:
        y_train = pickle.load(f)
    max_tracks = scales['max_tracks']
else:
    print("Loading data...", flush=True)
    with open("data/{}/x_train".format(data_dir), "rb") as f:
        x_train = pickle.load(f)
    # with open("data/{}/y_train".format(data_dir), "rb") as f:
    #     y_train = pickle.load(f)
    
    #tracks = [[track for track in line['live_tracks']['tracks'] if (track['vRel'] + line['v_ego'] > 1.34112) or (line['status'] and line['v_ego'] < 8.9408) or (line['v_ego'] < 8.9408)] for line in x_train] # remove tracks under 3 mph if no lead and above 20 mph
    tracks = [line['live_tracks']['tracks'] for line in x_train] # remove tracks under 3 mph if no lead and above 20 mph
    max_tracks = max([len(i) for i in tracks])  # max number of tracks in all samples
    
    # get relevant training car data to normalize
    car_data = [[line['v_ego'], line['steer_angle'], line['steer_rate'], line['a_lead'], line['left_blinker'], line['right_blinker'], line['status']] for line in x_train]
    
    print("Normalizing data...", flush=True)  # normalizes track dicts into [yRel, dRel, vRel trackStatus (0/1)] lists for training
    tracks_normalized, car_data_normalized, scales = normX(tracks, car_data)  # normalizes data and adds blinkers
    scales['max_tracks'] = max_tracks
    
    # print("Predicting brake data...", flush=True)
    # pos_preds = 0
    # neg_preds = 0
    # brake_preds = []
    # for idx, i in enumerate(x_train):  # use brake model to predict what the brake value is from v_ego and a_ego
    #     if y_train[idx] < 0.0:  # if brake sample
    #         to_pred = [interp_fast(i['v_ego'], brake_scales['v_ego_scale']), interp_fast(i['a_ego'], brake_scales['a_ego_scale'])]
    #         predicted_brake = interp_fast(brake_model.predict([[to_pred]])[0][0], [0, 1], [-1, 1])
    #         if predicted_brake < 0.0: # if prediction is to accel, default to coast (might want to choose arbitrary brake value)
    #             neg_preds += 1
    #             brake_preds.append(predicted_brake)
    #             y_train[idx] = predicted_brake #(predicted_brake - 0.02) * 1.05  # increase predicted brake to add weight
    #         else:
    #             pos_preds += 1
    #             y_train[idx] = 0.0
    
    # print('Of {} predictions, {} were incorrectly positive while {} were correctly negative.'.format(pos_preds + neg_preds, pos_preds, neg_preds))
    # print('The average brake prediction was {}, max {} and min {}'.format(sum(brake_preds) / len(brake_preds), min(brake_preds), max(brake_preds)))
    
    all_a = [i['a_ego'] for i in x_train]
    scales['a_ego'] = [min(all_a), max(all_a)]
    y_train = np.array([interp_fast(i['a_ego'], scales['a_ego'], [0, 1]) for i in x_train])
    # y_train = np.array([interp_fast(i, [-1, 1], [0, 1]) for i in y_train])
    
    with open("data/{}/x_train_normalized".format(data_dir), "wb") as f:
        pickle.dump([tracks_normalized, car_data_normalized, scales], f)
    
    with open("data/{}/y_train_normalized".format(data_dir), "wb") as f:
        pickle.dump(y_train, f)
#print(''+1)

#Format data
print("Sorting tracks...")
tracks_sorted = [sorted(line, key=lambda track: track[0]) for line in tracks_normalized]  # sort tracks by yRel

# pad tracks to max_tracks length so the shape is correct for training (keeps data in center of pad)
tracks_padded = [line if len(line) == max_tracks else pad_tracks(line, max_tracks) for line in tracks_sorted]  # tracks_sorted

# flatten tracks to 1d array
flat_tracks = [[item for sublist in sample for item in sublist] for sample in tracks_padded]

# combine into one list
x_train = np.array([i[0] + i[1] for i in zip(car_data_normalized, flat_tracks)])

#y_train = np.array([i if i >= 0 else 0.0 for i in y_train])  # pick some constant arbitrary negative value so we know when to warn user

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.01)
print(x_train.shape)

'''plt.clf()
secx = x_train[20000:20000+200]
secy = y_train[20000:20000+200]
x = range(len(secx))
y = [i['a_ego'] for i in secx]
y2 = secy
y3 = [i['v_ego']/30 for i in secx]
plt.plot(x, y, label='a_ego')
plt.plot(x, y2, label='gas')
plt.plot(x, y3, label='v_ego')

plt.legend()'''

try:
    os.mkdir("models/h5_models/{}".format(model_name))
except:
    pass

opt = keras.optimizers.Adam(lr=0.00055)#, decay=1.75e-4)
opt = keras.optimizers.Adadelta() #lr=.000375)
#opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
#opt = keras.optimizers.RMSprop(lr=0.00055, decay=1e-5)
#opt = keras.optimizers.Adagrad(lr=0.00025)
#opt = keras.optimizers.Adagrad()
#opt = 'adam'

#opt = 'rmsprop'
#opt = keras.optimizers.Adadelta()

layer_num = 6
nodes = 368
a_function = "relu"

model = Sequential()
model.add(Dense(x_train.shape[1], activation=a_function, input_shape=(x_train.shape[1:])))

for i in range(layer_num):
    model.add(Dense(nodes, activation=a_function))
    #model.add(tf.nn.dropout())
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
model.fit(x_train, y_train, shuffle=True, batch_size=256, epochs=8000, validation_data=(x_test, y_test))
#model = load_model("models/h5_models/{}.h5".format('live_tracksv6'))

#print("Gas/brake spread: {}".format(sum([model.predict([[[random.uniform(0,1) for i in range(4)]]])[0][0] for i in range(10000)])/10000)) # should be as close as possible to 0.5

seq_len = 100
plt.clf()
rand_start = random.randint(0, len(x_test) - seq_len)
x = range(seq_len)
y = [interp_fast(i, [0, 1], scales['a_ego']) for i in y_test[rand_start:rand_start+seq_len]]
y2 = [interp_fast(model.predict([[i]])[0][0], [0, 1], scales['a_ego']) for i in x_test[rand_start:rand_start+seq_len]]
plt.plot(x, y, label='ground truth')
plt.plot(x, y2, label='prediction')
plt.legend()
plt.show()


preds = []
for idx, i in enumerate(x_test):
    pred = model.predict([[i]])[0][0]
    preds.append(abs(interp_fast(pred, [0, 1], scales['a_ego']) - interp_fast(y_test[idx], [0, 1], scales['a_ego'])))

print("Test accuracy: {}".format(interp_fast(sum(preds) / len(preds), [0, max([abs(i) for i in scales['a_ego']])], [1, 0])))

'''for c in np.where(y_test==interp_fast(0.0, scales['a_ego'], [0, 1]))[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], scales['a_ego'])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], scales['a_ego'])))
    print()

for c in np.where(y_test>interp_fast(0.0, scales['a_ego'], [0, 1]))[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], scales['a_ego'])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], scales['a_ego'])))
    print()

for c in np.where(y_test<interp_fast(0.0, scales['a_ego'], [0, 1]))[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], scales['a_ego'])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], scales['a_ego'])))
    print()'''

'''preds = []
for idx, i in enumerate(x_train):
    preds.append(abs(model.predict([[i]])[0][0] - y_train[idx]))

print("Train accuracy: {}".format(1 - sum(preds) / len(preds)))'''

def feature_importance():
    #input_num = x_train.shape[1] - len(car_data[0])
    inputs = ['v_ego', 'steer_angle', 'steer_rate', 'a_lead', 'left_blinker', 'right_blinker', 'live_tracks']
    base = np.zeros(x_train.shape[1])
    base = model.predict([[base]])[0][0]
    preds = {}
    for idx, i in enumerate(inputs):
        a = np.zeros(x_train.shape[1])
        if i != 'live_tracks':
            np.put(a, idx, 1)
        else:
            np.put(a, range(len(inputs)+1, x_train.shape[1]), 1)
        preds[i] = abs(model.predict([[a]])[0][0] - base)
    
    plt.figure(2)
    plt.clf()
    [plt.bar(idx, preds[i], label=i) for idx, i in enumerate(preds)]
    [plt.text(idx, preds[i]+.007, str(round(preds[i], 5)), ha='center') for idx, i in enumerate(preds)]
    plt.xticks(range(0,len(inputs)), inputs)
    plt.title('Feature importance (difference from zero baseline)')
    plt.ylim(0, 1)
    plt.show()

def save_model(model_name=model_name):
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
#save_model()