import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten, Input
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
import shutil
import functools
import operator
import os
from keras.models import Model
from keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'
os.chdir("C:/Git/dynamic-follow-tf-v2/multinput")


samples = 500
a = np.random.uniform(low=0.0, high=1.0, size=(samples,2))
x_train1 = np.reshape(np.take(a, indices=0, axis=1), (samples, 1))
x_train2 = np.reshape(np.take(a, indices=1, axis=1), (samples, 1))
y_train = np.reshape(x_train1 + x_train2, (-1))

input_a = Input(shape=(1,), name='input_a')

x = Dense(16, activation='relu')(input_a)
output_a = Dense(16, activation='relu')(x)

input_b = Input(shape=(1,), name='input_b')
x = Dense(16, activation='relu')(input_b)
output_b = Dense(16, activation='relu')(x)
x = keras.layers.concatenate([output_a, output_b])

output_c = Dense(1)(x)

#model = Model(inputs=input_a, outputs=main_output)
model = Model(inputs=[input_a, input_b], outputs=output_c)
plot_model(model, to_file='model.png')
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])
model.fit([x_train1, x_train2], y_train, epochs=100)  # starts training