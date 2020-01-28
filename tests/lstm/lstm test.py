from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import keras

#LSTM format: (number_of_sequences, number_of_steps,features)

# return training data
def get_train():
    X=array([[[51, 50], [50, 49], [49, 48], [48, 47], [47, 46]], [[49, 48], [48, 47], [47, 46], [46, 45], [45, 44]]])
    y=array([[46, 45], [44, 43]])
    return X, y

opt = keras.optimizers.Adadelta()

X,y = get_train()

# define model
model = Sequential()
model.add(LSTM(800, input_shape=X.shape[1:]))
model.add(Dense(2, activation='linear'))
# compile model
model.compile(loss='mse', optimizer=opt)
# fit model
model.fit(X, y, epochs=200, shuffle=False)

pred=model.predict([[X[0]]])
nextseq=np.concatenate([list(X[0][1:]), list(pred)])
pred2=model.predict([[nextseq]])
print(pred)
print(pred2)
