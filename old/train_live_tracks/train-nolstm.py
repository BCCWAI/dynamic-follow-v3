'''from numpy.random import seed
seed(255)
from tensorflow import set_random_seed
set_random_seed(255)'''
import os
import tensorflow as tf
from keras.models import Sequential
from keras.backend.tensorflow_backend import set_session
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten, PReLU, ELU, LeakyReLU
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
# import load_brake_pred_model as brake_wrapper
# from keras.callbacks.tensorboard_v1 import TensorBoard

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = .4
# set_session(tf.Session(config=config))


# brake_model, brake_scales = brake_wrapper.get_brake_pred_model()


def interp_fast(x, xp, fp=[0, 1], ext=False):  # extrapolates above range when ext is True
    interped = (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]
    return interped if ext else min(max(min(fp), interped), max(fp))


def pad_tracks(tracks, max_tracks):
    to_add = max_tracks - len(tracks)
    to_add_left = to_add - (to_add // 2)
    to_add_right = to_add - to_add_left
    to_pad = [[0, 0, 0]]
    return tracks + (to_add * to_pad)  # these two have little to no difference
    # return (to_pad * to_add_left) + tracks + (to_pad * to_add_right)


os.chdir("C:/Git/dynamic-follow-tf-v2")
data_dir = "live_tracks"
norm_dir = "data/{}/normalized"
model_name = "live_tracks"

try:
    shutil.rmtree("models/h5_models/{}".format(model_name))
except:
    pass


def feature_importance():
    # input_num = x_train.shape[1] - len(car_data[0])
    inputs = ['v_ego', 'steer_angle', 'steer_rate', 'a_lead', 'left_blinker', 'right_blinker', 'live_tracks']
    base = np.zeros(x_train.shape[1])
    base = model.predict([[base]])[0][0]
    preds = {}
    for idx, i in enumerate(inputs):
        a = np.zeros(x_train.shape[1])
        if i != 'live_tracks':
            np.put(a, idx, 1)
        else:
            np.put(a, range(len(inputs) + 1, x_train.shape[1]), 1)
        preds[i] = abs(model.predict([[a]])[0][0] - base)

    plt.figure(2)
    plt.clf()
    [plt.bar(idx, preds[i], label=i) for idx, i in enumerate(preds)]
    [plt.text(idx, preds[i] + .007, str(round(preds[i], 5)), ha='center') for idx, i in enumerate(preds)]
    plt.xticks(range(0, len(inputs)), inputs)
    plt.title('Feature importance (difference from zero baseline)')
    plt.ylim(0, 1)
    plt.pause(0.01)
    plt.show()


def show_coast(to_display=100):
    plt.figure(0)
    plt.clf()
    plt.title('coast samples: predicted vs ground')
    find = .0
    found = [idx for idx, i in enumerate(y_test) if i == find and x_test[idx][0] > .62]  # and going above 40 mph
    found = np.random.choice(found, to_display)
    ground = [interp_fast(y_test[i], [0, 1], [-1, 1]) for i in found]
    pred = [interp_fast(model.predict([[x_test[i]]])[0][0], [0, 1], [-1, 1]) for i in found]
    plt.plot(range(len(found)), ground, label='ground truth')
    plt.scatter(range(len(ground)), pred, label='prediction', s=20)
    plt.ylim(-.2, 1.0)
    plt.legend()
    plt.pause(0.001)
    plt.show()

class Visualize(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # feature_importance()
        show_coast()
        pass

if os.path.exists("data/{}/x_train_normalized".format(data_dir)):
    print('Loading normalized data...', flush=True)
    with open("data/{}/x_train_normalized".format(data_dir), "rb") as f:
        tracks_normalized, car_data_normalized, scales = pickle.load(f)
    with open("data/{}/y_train_normalized".format(data_dir), "rb") as f:
        y_train = pickle.load(f)
else:
    print("Loading data...", flush=True)
    with open("data/{}/x_train".format(data_dir), "rb") as f:
        x_train = pickle.load(f)
    with open("data/{}/y_train".format(data_dir), "rb") as f:
        y_train = pickle.load(f)

    remove_blinkers = False
    if remove_blinkers:
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if True not in [x['left_blinker'], x['right_blinker']]])  # filter samples with turn signals
        x_train, y_train = map(list, [x_train, y_train])

    only_leads = False
    if only_leads:
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if x['lead_status']])
        x_train, y_train = map(list, [x_train, y_train])

    filter_low_speeds = False
    if filter_low_speeds:
        mph_to_ms = 0.44704
        min_speed = 55 * mph_to_ms
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if x['v_ego'] >= min_speed])
        x_train, y_train = map(list, [x_train, y_train])

    data_filter = "all"  # can be "gas", "brake", or "all" to do nothing
    predict_brake = False

    if data_filter == "gas":
        print("Filtering out brake samples...")
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if y >= 0.0])  # keep gas or coast samples
        x_train, y_train = map(list, [x_train, y_train])
    elif data_filter == "brake":
        print("Filtering out gas samples...")
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if y <= 0.0])  # keep only brake or coast samples
        x_train, y_train = map(list, [x_train, y_train])

    filter_slow_tracks = False
    if filter_slow_tracks:
        tracks = [[track for track in line['live_tracks']['tracks'] if track['vRel'] + line['v_ego'] >= 8.9408] for line in x_train]
    else:
        tracks = [line['live_tracks']['tracks'] for line in x_train]

    # tracks = [[track for track in line['live_tracks']['tracks'] if (track['vRel'] + line['v_ego']) >= 0 and ((line['v_ego'] >= 8.9408 and track['vRel'] + line['v_ego'] > 2.2352) or (line['v_ego'] < 8.9408)) and abs(track['yRel']) < 4.45] for line in x_train]

    # tracks = [[track for track in line['live_tracks']['tracks'] if
    #            # keep track if v_ego above 25 mph and track vel is above 5 mph OR if v_ego is less than 25 mph
    #            ((line['v_ego'] >= 11.176 and track['vRel'] + line['v_ego'] > 2.2352) or
    #             line['v_ego'] < 11.176)] for line in x_train]

    # tracks = [[track for track in line['live_tracks']['tracks'] if (track['vRel'] + line['v_ego'] > 1.34112) or (line['status'] and line['v_ego'] < 8.9408) or (line['v_ego'] < 8.9408)] for line in x_train] # remove tracks under 3 mph if no lead and above 20 mph
    
    # get relevant training car data to normalize
    car_data = [[line['v_ego'], line['steer_angle'], line['steer_rate'], line['a_lead'], line['left_blinker'], line['right_blinker'], line['lead_status'], line['x_lead'], line['v_lead']] for line in x_train]
    # car_data = [[line['v_ego'], line['steer_angle'], line['steer_rate'], line['a_lead'], line['left_blinker'], line['right_blinker'], line['status']] for line in x_train]
    
    print("Normalizing data...", flush=True)  # normalizes track dicts into [yRel, dRel, vRel trackStatus (0/1)] lists for training
    tracks_normalized, car_data_normalized, scales = normX(tracks, car_data)  # normalizes data and adds blinkers
    scales['max_tracks'] = max([len(i) for i in tracks])  # max number of tracks in all samples

    if data_filter in ["brake", "all"] and predict_brake:
        print("Predicting brake samples...", flush=True)
        pos_preds = 0
        neg_preds = 0
        #brake_preds = []
        brake_indices = [idx for idx, y_sample in enumerate(y_train) if y_sample < 0.0]
        brake_samples = [x_train[idx] for idx in brake_indices]
        to_pred = [[interp_fast(i['v_ego'], brake_scales['v_ego_scale']), interp_fast(i['a_ego'], brake_scales['a_ego_scale'])] for i in brake_samples]
        brake_preds = brake_model.predict([to_pred]).reshape(1, -1)[0]
        for idx, predicted_brake in enumerate(brake_preds):
            if predicted_brake < 0.0:
                neg_preds += 1
                y_train[brake_indices[idx]] = predicted_brake * 1.1
            else:
                pos_preds += 1
                y_train[brake_indices[idx]] = -0.15

        print('Of {} predictions, {} were incorrectly positive while {} were correctly negative.'.format(pos_preds + neg_preds, pos_preds, neg_preds))
        #print('The average brake prediction was {}, max {} and min {}'.format(sum(brake_preds) / len(brake_preds), min(brake_preds), max(brake_preds)))

    #scales['gas'] = [min(y_train), max(y_train)]
    y_train = np.array(y_train)
    print('Dumping normalized data...', flush=True)
    with open("data/{}/x_train_normalized".format(data_dir), "wb") as f:
        pickle.dump([tracks_normalized, car_data_normalized, scales], f)
    with open("data/{}/y_train_normalized".format(data_dir), "wb") as f:
        pickle.dump(y_train, f)

#print(''+1)
#Format data
print("Sorting tracks...")  # sorting seems to make a difference
#tracks_sorted = [sorted(line, key=lambda track: track[0]) for line in tracks_normalized]  # sort tracks by yRel

# pad tracks to max_tracks length so the shape is correct for training (keeps data in center of pad)
tracks_padded = [line if len(line) == scales['max_tracks'] else pad_tracks(line, scales['max_tracks']) for line in
                 tracks_normalized]  # tracks_sorted

# flatten tracks to 1d array
flat_tracks = [[item for sublist in sample for item in sublist] for sample in tracks_padded]

# combine into one list
x_train = np.array([car_dat + fl_tr for car_dat, fl_tr in zip(car_data_normalized, flat_tracks)])

# x_train = np.array(car_data_normalized)  # todo; experiment with this

# y_train = np.array([i if i >= 0 else 0.0 for i in y_train])  # pick some constant arbitrary negative value so we know when to warn user

# y_train = np.array([interp_fast(i, [-1, 1], [0, 1]) for i in y_train])  # EXPERIMENT WITH THIS
# y_train = np.array(y_train)


# scales['gas'] = [min(y_train), max(y_train)]
# y_train = np.interp(y_train, scales['gas'], [0, 1])
# y_train = np.interp(y_train, [-1, 1], [0, 1])  # this is the best performing model architecture


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
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

# try:
#     os.mkdir("models/h5_models/{}".format(model_name))
# except:
#     pass

#opt = keras.optimizers.Adam(lr=0.001)
# opt = keras.optimizers.Adadelta() #lr=.000375)
# opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
# opt = keras.optimizers.RMSprop(lr=0.00005)#, decay=1e-5)
# opt = keras.optimizers.Adagrad(lr=0.00025)
# opt = keras.optimizers.Adagrad()
opt = 'adam'

#opt = 'rmsprop'
# opt = keras.optimizers.Adadelta()

layer_num = 6
nodes = 346
a_function = "relu"

model = Sequential()
model.add(Dense(x_train.shape[1] + 1, activation=None, input_shape=(x_train.shape[1:])))
model.add(Dense(256, activation=a_function))
model.add(Dense(128, activation=a_function))
model.add(Dense(64, activation=a_function))
model.add(Dense(64, activation=a_function))
model.add(Dense(1, activation='tanh'))

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])

# tensorboard = TensorBoard(log_dir="C:/Git/dynamic-follow-tf-v2/train_model/logs/{}".format("final model"))
# callbacks = [tensorboard]
model.fit(x_train, y_train,
          shuffle=True,
          batch_size=256,
          epochs=10000,
          validation_data=(x_test, y_test))
          # callbacks=callbacks)

# model = load_model("models/h5_models/{}.h5".format('live_tracksvHOLDENONLYLEADS'))

#print("Gas/brake spread: {}".format(sum([model.predict([[[random.uniform(0,1) for i in range(4)]]])[0][0] for i in range(10000)])/10000)) # should be as close as possible to 0.5

seq_len = 100
plt.clf()
rand_start = random.randint(0, len(x_test) - seq_len)
x = range(seq_len)
y = y_test[rand_start:rand_start+seq_len]
y2 = [model.predict([[i]])[0][0] for i in x_test[rand_start:rand_start+seq_len]]
plt.title("random samples")
plt.plot(x, y, label='ground truth')
plt.plot(x, y2, label='prediction')
plt.legend()
plt.pause(0.01)
plt.show()

def visualize(to_display=1000):
    try:
        plt.figure(0)
        plt.clf()
        plt.title('coast samples: predicted vs ground')
        find = .5
        found = [idx for idx, i in enumerate(y_test) if i == find]
        found = np.random.choice(found, to_display)
        ground = [y_test[i] for i in found]
        pred = [model.predict([[x_test[i]]])[0][0] for i in found]
        plt.plot(range(len(found)), ground, label='ground truth')
        plt.plot(range(len(ground)), pred, label='prediction')
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.show()
    except:
        pass

    try:
        plt.figure(1)
        plt.clf()
        plt.title('medium acceleration samples: predicted vs ground')
        find = .625
        found = [idx for idx, i in enumerate(y_test) if abs(i - find) < .001]
        found = np.random.choice(found, to_display)
        ground = [y_test[i] for i in found]
        pred = [model.predict([[x_test[i]]])[0][0] for i in found]
        plt.plot(range(len(found)), ground, label='ground truth')
        plt.plot(range(len(ground)), pred, label='prediction')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.show()
    except:
        pass

    try:
        plt.figure(2)
        plt.clf()
        plt.title('heavy acceleration samples: predicted vs ground')
        find = .75
        found = [idx for idx, i in enumerate(y_test) if abs(i - find) < .001]
        found = np.random.choice(found, to_display)
        ground = [y_test[i] for i in found]
        pred = [model.predict([[x_test[i]]])[0][0] for i in found]
        plt.plot(range(len(found)), ground, label='ground truth')
        plt.plot(range(len(ground)), pred, label='prediction')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.show()
    except:
        pass

    try:
        plt.figure(3)
        plt.clf()
        plt.title('medium brake samples: predicted vs ground')
        find = 0.4
        found = [idx for idx, i in enumerate(y_test) if abs(i - find) < .001]
        found = np.random.choice(found, to_display)
        ground = [y_test[i] for i in found]
        pred = [model.predict([[x_test[i]]])[0][0] for i in found]
        plt.plot(range(len(found)), ground, label='ground truth')
        plt.plot(range(len(ground)), pred, label='prediction')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.show()
    except:
        pass

    try:
        plt.figure(3)
        plt.clf()
        plt.title('hard brake samples: predicted vs ground')
        find = 0.25
        found = [idx for idx, i in enumerate(y_test) if abs(i - find) < .1]
        found = np.random.choice(found, to_display)
        ground = [y_test[i] for i in found]
        pred = [model.predict([[x_test[i]]])[0][0] for i in found]
        plt.plot(range(len(found)), ground, label='ground truth')
        plt.plot(range(len(ground)), pred, label='prediction')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.show()
    except:
        pass


preds = model.predict([x_test]).reshape(1, -1)
diffs = [abs(pred - ground) for pred, ground in zip(preds[0], y_test)]

print("Test accuracy: {}".format(interp_fast(sum(diffs) / len(diffs), [0, 1], [1, 0], ext=True)))

for i in range(20):
    c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict([x_test[c]])[0][0]))
    print()

for c in np.where(y_test==0.5)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], [-1, 1])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], [-1, 1])))
    print()


def coast_test():
    coast_samples = np.where(y_test == 0.5)[0]
    coast_predictions = model.predict(x_test[(coast_samples)])
    num_invalid = 0
    for i in coast_predictions:
        if abs((i[0] - 0.5) * 2.0 >= .05):
            num_invalid += 1
    print('Out of {} samples, {} predictions were invalid (above 0.01 threshold)'.format(len(coast_samples), num_invalid))
    print('Percentage: {}'.format(1 - num_invalid / len(coast_samples)))



'''for c in np.where(y_test>0.5)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], [-1, 1])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], [-1, 1])))
    print()

for c in np.where(y_test<0.5)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(interp_fast(y_test[c], [0, 1], [-1, 1])))
    print('Prediction: {}'.format(interp_fast(model.predict([[x_test[c]]])[0][0], [0, 1], [-1, 1])))
    print()'''

'''preds = []
for idx, i in enumerate(x_train):
    preds.append(abs(model.predict([[i]])[0][0] - y_train[idx]))

print("Train accuracy: {}".format(1 - sum(preds) / len(preds)))'''

def save_model(model_name=model_name):
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
#save_model()