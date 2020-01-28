import os
import json
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, BatchNormalization, LeakyReLU, Flatten, TimeDistributed, CuDNNGRU, RNN, SimpleRNN, GRU
import numpy as np
import random
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
import time
from normalizer import norm
import matplotlib.pyplot as plt
import pickle
import sys
from sklearn import preprocessing
import itertools
from sklearn.model_selection import train_test_split

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class Visualize(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0 or 1==1:
            preds = []
            grounds = []
            x = list(range(len(test_samples)))
            for i in test_samples:
                preds.append(model.predict([[x_test[i]]])[0][0])
                grounds.append(y_test[i])
            
            plt.clf()
            plt.plot(x, grounds, label='ground-truth')#, marker='o')
            plt.plot(x, preds, label='prediction-{}'.format(epoch))#, marker='o')
            plt.title("test data (unseen)")
            plt.legend()
            plt.savefig("models/h5_models/{}/0-{}-epoch-{}.png".format(model_name, model_name, epoch))
            plt.pause(.1)
            model.save("models/h5_models/{}/{}-epoch-{}.h5".format(model_name, model_name, epoch))

#v_ego, v_lead, d_lead
data_dir = "LSTM"
os.chdir("C:/Git/dynamic-follow-tf")

norm_dir = "data/{}/normalized.npy"

model_name = "LSTM"

'''with open("data/{}/x_train".format(data_dir), "rb") as f:
    x_train = pickle.load(f)

with open("data/{}/y_train".format(data_dir), "rb") as f:
    y_train = pickle.load(f)'''

samples_to_use = 7000000
if samples_to_use != 'all':
    y_train = np.load("data/{}/y_train.npy".format(data_dir))[:samples_to_use]
else:
    y_train = np.load("data/{}/y_train.npy".format(data_dir))
    print(len(y_train))
is_array = False

if not os.path.exists(norm_dir.format(data_dir)):
    if samples_to_use != 'all':
        x_train = np.load("data/{}/x_train.npy".format(data_dir))[:samples_to_use]
    else:
        x_train = np.load("data/{}/x_train.npy".format(data_dir))
    print("Normalizing...", flush=True)
    normalized = norm(x_train)
    x_train = normalized['normalized']
    scales = normalized['scales']
    print("Dumping normalization...", flush=True)
    np.save(norm_dir.format(data_dir), x_train)
    with open('data/LSTM/scales', "wb") as f:
        pickle.dump(normalized['scales'], f)
    #with open(norm_dir.format(data_dir), "wb") as f:
        #pickle.dump(normalized, f)
else:
    is_array = True
    print("Loading normalized data...", flush=True)
    x_train = np.load(norm_dir.format(data_dir))
    with open('data/LSTM/scales', "rb") as f:
        scales = pickle.load(f)
    print('Loaded!', flush=True)
    #with open(norm_dir.format(data_dir), "rb") as f:
        #normalized = pickle.load(f)

print(len(x_train))
#scales = normalized['scales']
#x_train = normalized['normalized']
y_train = np.array([np.interp(i, [-1, 1], [0, 1]) for i in y_train])

flatten = True
if flatten:
    print('Flattening...', flush=True)
    #x_train = np.array([[inner for outer in sample for inner in outer] for sample in x_train])
    x_train = np.array([i.flatten() for i in x_train])  # whole lot faster lul
elif not is_array:
    x_train = np.array(x_train)
print(x_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

#random_choices = []
num_test = 600
num_test = min(len(x_test), num_test)

test_samples = random.sample(range(len(x_test)), num_test)
test_samples = sorted(list(zip([y_test[i] for i in test_samples], test_samples)))
test_samples = [y for x, y in test_samples] # sort from lowest to highest for visualization

opt = keras.optimizers.Adam(lr=0.0001)#, decay=1e-6)
opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.RMSprop(0.001)
#opt = keras.optimizers.Adagrad(lr=0.00001)
opt = 'adam'
#opt = 'rmsprop'
#opt = keras.optimizers.SGD(lr=0.01, decay=1e-8, momentum=0.9, nesterov=True)

layers = 6
nodes = 186

model_type = 'dense' # lstm, dense, or gru

model = Sequential()
if model_type.lower() == 'lstm':
    to_sub = 2
    model.add(CuDNNLSTM(nodes, input_shape=(x_train.shape[1:]), return_sequences=True))
elif model_type.lower() == 'dense':
    to_sub = 1
    model.add(Dense(nodes, activation="relu", input_shape=(x_train.shape[1:])))
    #model.add(Dropout(0.1))
#model.add(CuDNNGRU(nodes, return_sequences=True, input_shape=(x_train.shape[1:])))
#model.add(SimpleRNN(128, activation='tanh', return_sequences=False, input_shape=(x_train.shape[1:])))
#model.add(Dropout(.05))
for i in range(layers - to_sub):
    #model.add(Dense(nodes, activation="relu"))
    #model.add(Dropout(.05))
    if model_type.lower() == 'lstm':
        model.add(CuDNNLSTM(nodes, return_sequences=True))
    #model.add(CuDNNGRU(nodes, return_sequences=True))
    elif model_type.lower() == 'dense':
        model.add(Dense(nodes, activation="relu"))
        #model.add(Dropout(0.1))
    #model.add(SimpleRNN(64, activation='tanh'))
#model.add(Dense(128, activation='relu'))
#model.add(Permute((2,1), input_shape=(10, 5)))
#model.add(CuDNNGRU(nodes, return_sequences=False))
if model_type.lower() == 'lstm':
    model.add(CuDNNLSTM(nodes, return_sequences=False))

model.add(Dense(1, activation='linear'))


model.compile(loss='mean_squared_error', optimizer=opt)
#tensorboard = TensorBoard(log_dir="logs/test-{}".format("30epoch"))
callback_list = [Visualize()]
model.fit(x_train, y_train, shuffle=True, batch_size=256, epochs=200, validation_data=(x_test, y_test), callbacks=callback_list)


def get_acc():
    accs = []
    for idx, i in enumerate(x_test[:20000]):
        pred = model.predict([[i]])[0][0]
        accs.append(abs(pred - y_test[idx]))
    print('Test accuracy: {}'.format(1 - sum(accs) / len(accs)))
    
    '''accs = []
    for idx, i in enumerate(x_train):
        pred = model.predict([[i]])[0][0]
        accs.append(abs(pred - y_train[idx]))
    print('Train accuracy: {}'.format(1 - sum(accs) / len(accs)))'''

get_acc()
    
for i in range(10):
    rand = random.randint(0, len(x_train))
    pred = model.predict([[x_train[rand]]])[0][0]
    print("Ground truth: {}".format(y_train[rand]))
    print("Prediction: {}\n".format(pred))

save_model = False
if save_model:
    model.save("models/h5_models/"+model_name+".h5")
    print("Saved model!")