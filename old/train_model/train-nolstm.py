'''from numpy.random import seed
seed(255)
from tensorflow import set_random_seed
set_random_seed(255)'''
import os
import tensorflow as tf
from keras.models import Sequential
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
import load_brake_pred_model_corolla as brake_wrapper
from keras.callbacks.tensorboard_v1 import TensorBoard


brake_model, brake_scales = brake_wrapper.get_brake_pred_model()


def interp_fast(x, xp, fp=[0, 1], ext=False):  # extrapolates above range when ext is True
    interped = (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]
    return interped if ext else min(max(min(fp), interped), max(fp))


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
    inputs = ['v_ego', 'v_lead', 'x_lead', 'a_lead']
    base = np.zeros(x_train.shape[1])
    base = model.predict([[base]])[0][0]
    preds = {}
    for idx, i in enumerate(inputs):
        a = np.zeros(x_train.shape[1])
        np.put(a, idx, 1)
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


def show_coast(to_display=200):
    plt.figure(0)
    plt.clf()
    plt.title('coast samples: predicted vs ground')
    find = .5
    found = [idx for idx, i in enumerate(y_test) if i == find and x_test[idx][0] > .62]  # and going above 40 mph
    found = np.random.choice(found, to_display)
    ground = [interp_fast(y_test[i], [0, 1], [-1, 1]) for i in found]
    pred = [interp_fast(model.predict([[x_test[i]]])[0][0], [0, 1], [-1, 1]) for i in found]
    plt.plot(range(len(found)), ground, label='ground truth')
    plt.scatter(range(len(ground)), pred, label='prediction', s=20)
    plt.ylim(-1.0, 1.0)
    plt.legend()
    plt.pause(0.01)
    plt.show()

class Visualize(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # feature_importance()
        # show_coast()
        pass


if os.path.exists("data/{}/x_train_normalized".format(data_dir)):
    print('Loading normalized data...', flush=True)
    with open("data/{}/x_train_normalized".format(data_dir), "rb") as f:
        car_data_normalized, scales = pickle.load(f)
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

    model_inputs = ['v_ego', 'v_lead', 'x_lead', 'a_lead']
    
    print("Normalizing data...", flush=True)  # normalizes track dicts into [yRel, dRel, vRel trackStatus (0/1)] lists for training
    car_data_normalized, scales = normX(x_train, model_inputs)  # normalizes data and adds blinkers

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
                y_train[brake_indices[idx]] = predicted_brake * 1.275
            else:
                pos_preds += 1
                y_train[brake_indices[idx]] = -0.15

        print('Of {} predictions, {} were incorrectly positive while {} were correctly negative.'.format(pos_preds + neg_preds, pos_preds, neg_preds))
        #print('The average brake prediction was {}, max {} and min {}'.format(sum(brake_preds) / len(brake_preds), min(brake_preds), max(brake_preds)))

    #scales['gas'] = [min(y_train), max(y_train)]

    print('Dumping normalized data...', flush=True)
    with open("data/{}/x_train_normalized".format(data_dir), "wb") as f:
        pickle.dump([car_data_normalized, scales], f)
    with open("data/{}/y_train_normalized".format(data_dir), "wb") as f:
        pickle.dump(y_train, f)

x_train = np.array(car_data_normalized)


scales['gas'] = [min(y_train), max(y_train)]
# y_train = np.interp(y_train, scales['gas'], [0, 1])
y_train = np.interp(y_train, [-1, 1], [0, 1])  # this is the best performing model architecture


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)
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
model.add(Dense(256, activation=a_function))
model.add(Dense(64, activation=a_function))
model.add(Dense(32, activation=a_function))
# for i in range(layer_num):
#     model.add(Dense(nodes, activation=a_function))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])

tensorboard = TensorBoard(log_dir="C:/Git/dynamic-follow-tf-v2/train_model/logs/{}".format("final model"))
callbacks = [tensorboard]
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
    print('Prediction: {}'.format(model.predict([[x_test[c]]])[0][0]))
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
        if abs((i[0] - 0.5) * 2.0 >= .02):
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