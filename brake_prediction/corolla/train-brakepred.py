import json
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten, CuDNNGRU
import numpy as np
import random
from normalizer_brakepred import normX
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
from sklearn.model_selection import train_test_split
import shutil
import functools
import operator
from keras.models import load_model

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .1
set_session(tf.Session(config=config))


def interp_fast(x, xp, fp=[0, 1], ext=False):  # extrapolates above range when ext is True
    interped = (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]
    return interped if ext else min(max(min(fp), interped), max(fp))

os.chdir("C:/Git/dynamic-follow-tf-v2")
data_dir = "brake_pred-Corolla"
norm_dir = "data/{}/normalized"
model_name = "brake_pred-Corolla"

print("Loading data...")
with open("data/{}/x_train".format(data_dir), "rb") as f:
    x_train = pickle.load(f)
with open("data/{}/y_train".format(data_dir), "rb") as f:
    y_train = pickle.load(f)

# print("Loading test data...")
# with open("data/live_tracks/x_train", "rb") as f:
#     x_train_nobrake = pickle.load(f)
# with open("data/live_tracks/y_train", "rb") as f:
#     y_train_nobrake = np.array(pickle.load(f))
# x_train, y_train = zip(*[i for i in zip(x_train, y_train) if i[1] <= 0])
# x_train, y_train = np.array(x_train), np.array(y_train)

# x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if (y < 0 and x[0] > x[1]) or y >= 0 or x[0] > 2])
# x_train, y_train = map(np.array, [x_train, y_train])

# for i in range(50):
#     c = random.choice(range(len(x_train)))
#     plt.clf()
#     plt.plot(x_train[c])
#     plt.title(y_train[c])
#     plt.pause(0.01)
#     input()
# raise Exception

x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if y <= 0.0])  # keep only brake or coast samples
x_train, y_train = map(np.array, [x_train, y_train])

print("Normalizing data...", flush=True)
x_train, v_ego_scale = normX(x_train)
x_train = np.array(x_train)

gas_scale = [min(y_train), max(y_train)]

y_train = np.interp(y_train, gas_scale, [0, 1])

with open("data/{}/scales".format(data_dir), "w") as f:
    json.dump({'v_ego': v_ego_scale, 'gas': gas_scale}, f)

use_dense = True
if not use_dense:
    x_train = np.array([seq.reshape(-1, 1) for seq in x_train])

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

#opt = keras.optimizers.Adam()
# opt = keras.optimizers.Adadelta(lr=.000375)
#opt = keras.optimizers.SGD(lr=0.01, momentum=0.9)
# opt = keras.optimizers.RMSprop(lr=0.00005)#, decay=1e-5)
# opt = keras.optimizers.Adagrad(lr=0.00025)
# opt = keras.optimizers.Adagrad()
opt = 'adam'

# opt = 'rmsprop'
# opt = keras.optimizers.Adadelta()

layer_num = 5
nodes = 64
a_function = "relu"
model = Sequential()
if use_dense:
    model.add(Dense(len(x_train[0]) + 1, input_shape=x_train.shape[1:]))
    model.add(Dense(128, activation=a_function))
    model.add(Dense(64, activation=a_function))
    model.add(Dense(64, activation=a_function))
else:
    model.add(CuDNNGRU(64, return_sequences=True))
    model.add(CuDNNGRU(32, return_sequences=True))
    model.add(CuDNNGRU(16))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])

model.fit(x_train,
          y_train,
          shuffle=True,
          batch_size=64,
          epochs=1000,
          validation_data=(x_test, y_test))
#model = load_model("models/h5_models/{}.h5".format(model_name))

#x_train_all = x_train_all[20000:25000]
#y_train_all = y_train_all[20000:25000]


def find_best_model(min_nodes=8, max_nodes=1024, steps=5, epochs=5, batch_size=512):
    acc_dict = {}
    for i in range(steps):
        nodes = round(interp_fast(i, [0, steps - 1], [min_nodes, max_nodes]))
        print("\nTesting {} nodes!\n".format(nodes))
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(nodes, activation="relu"))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        model.fit(x_train, y_train, shuffle=False, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), verbose=False)

        preds = model.predict(x_test).reshape(1, -1)[0]
        diffs = [abs(pred - ground) for pred, ground in zip(preds, y_test)]
        acc = interp_fast(sum(diffs) / len(diffs), [0, .25], [1, 0], ext=True)
        acc_dict[nodes] = acc

    sorted_acc = sorted(acc_dict, key=acc_dict.get, reverse=True)
    print("\n{}".format('\n'.join(["Nodes: {}, accuracy: {}%".format(nod, round(acc * 100, 5)) for nod, acc in zip(sorted_acc, [acc_dict[i] for i in sorted_acc])])))



#print("Gas/brake spread: {}".format(sum([model.predict([[[random.uniform(0,1) for i in range(4)]]])[0][0] for i in range(10000)])/10000)) # should be as close as possible to 0.5

preds = model.predict(x_test).reshape(1, -1)[0]
diffs = [abs(pred - ground) for pred, ground in zip(preds, y_test)]

print("Test accuracy: {}".format(interp_fast(sum(diffs) / len(diffs), [0, 1], [1, 0], ext=True)))


'''x = [50-i for i in range(50)]
y = [interp_fast(model.predict([[[interp_fast(i, scales['v_ego_scale']), interp_fast(-2, scales['a_ego_scale'])]]])[0][0], [0, 1], gas_scale) for i in x]
plt.plot(x, y)
plt.show()'''

x = range(50)
y = []
ground = []
while len(y) < 50:
    c = random.randrange(len(x_test))
    y.append(model.predict([[x_test[c]]])[0][0])
    ground.append(y_test[c])
plt.plot(x, y, label='pred')
plt.plot(x, ground, label='ground')
plt.title('train data')
plt.legend()
plt.show()

# x = range(50)
# y = []
# y_true = []
# while len(y) != 50:
#     c = random.randrange(len(x_train_nobrake))
#     if y_train_nobrake[c] < 0.0:
#         to_pred = [interp_fast(x_train_nobrake[c]['v_ego'], scales['v_ego_scale'], [0, 1]), interp_fast(x_train_nobrake[c]['a_ego'], scales['a_ego_scale'], [0, 1])]
#         y.append(model.predict([[to_pred]])[0][0])
#         y_true.append(y_train_nobrake[c])
# plt.plot(x,y, label='pred')
# plt.plot(x,y_true, label='ground')
# plt.title('live tracks data')
# plt.legend()
# plt.show()


for c in np.where(y_test < .5)[0][:20]:
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict([[x_test[c]]])[0][0]))
    print()

# for i in range(20):
#     c = random.randint(0, len(x_test))
#     print('Ground truth: {}'.format(y_test[c]))
#     print('Prediction: {}'.format(model.predict([[x_test[c]]])[0][0]))
#     print()


'''showed = 0
while showed <= 20:
    c = random.randint(0, len(x_train_nobrake))
    if x_train_nobrake[c]['v_ego'] > 8.9 and y_train_nobrake[c] >= 0.0:
        showed+=1
        print('Ground truth: {}'.format(y_train_nobrake[c]))
        to_pred = [interp_fast(x_train_nobrake[c]['v_ego'], scales['v_ego_scale'], [0, 1]), interp_fast(x_train_nobrake[c]['a_ego'], scales['a_ego_scale'], [0, 1])]
        print('Prediction: {}'.format(interp_fast(model.predict([[to_pred]])[0][0], [0, 1], gas_scale)))
        print()'''


save_model = True
tf_lite = False
if save_model:
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)