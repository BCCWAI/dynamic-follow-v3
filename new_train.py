import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten, MaxPooling1D
import numpy as np
import random
from normalizer import norm
from sklearn.preprocessing import normalize
import random
#np.set_printoptions(threshold=np.inf)

#v_ego, v_lead, d_lead
os.chdir("C:/Git/dynamic-follow-tf")

#x_train = np.asarray([[[1,2,3,4,5], [6,7,8,9,10]]])
x_train = np.asarray([[[20, 20], [20, 15]], [[20,20], [20,25]], [[20,20], [20,20]], [[50,50], [50,50]]])
y_train = np.asarray([-.2, .2, 0, 0])

opt = keras.optimizers.Adam(lr=.001, decay=.0001)

model = Sequential()
#model.add(Dense(2, activation="relu", input_shape=(2,5)))
model.add(LSTM(24, activation="relu", return_sequences=True, input_shape=(x_train.shape[1:])))
model.add(LSTM(24, activation="relu", return_sequences=True))
model.add(LSTM(24, activation="relu", return_sequences=True))
model.add(Flatten())
model.add(Dense(12, activation="relu"))
model.add(Dense(1))

model.compile(loss='mean_absolute_error', optimizer=opt, metrics=["mean_squared_error"])
model.fit(x_train, y_train, epochs=100)

print(model.predict(np.asarray([[[20, 20], [20, 20]]])))

'''data = [norm(23.74811363, v_ego_scale), norm(-0.26912481, a_ego_scale), norm(15.10309029, v_lead_scale), norm(55.72000122, x_lead_scale), norm(-0.31268027, a_lead_scale)] #should be -0.5
prediction=model.predict(np.asarray([data]))[0][0]
print((prediction - 0.5)*2.0)

#accur = list([list(i) for i in x_train])

accuracy=[]
for i in range(500):
    choice = random.randint(0, len(x_train - 1))
    real=y_train[choice]
    to_pred = list(list(x_train)[choice])
    pred = model.predict(np.asarray([to_pred]))[0][0]
    accuracy.append(abs(real-pred))
    #print("Real: "+str(real))
    #print("Prediction: "+str(pred))
    #print()
    
avg = sum(accuracy) / len(accuracy)
if NORM:
    print("Accuracy: "+ str(abs(avg-1)))
else:
    print("Accuracy: "+ str(np.interp(avg, [0, 1], [1, 0])))'''

#test_data = [[norm(15, v_ego_scale), norm(0, a_ego_scale), norm(15, v_lead_scale), norm(18, x_lead_scale), norm(0, a_lead_scale)]]

#print(model.predict(np.asarray(test_data)))

save_model = False
tf_lite = False
if save_model:
    model_name = "combined"
    model.save("models/h5_models/"+model_name+".h5")
    if tf_lite:
        # convert model to tflite:
        converter = tf.lite.TFLiteConverter.from_keras_model_file("models/h5_models/"+model_name+".h5")
        tflite_model = converter.convert()
        open("models/lite_models/"+model_name+".tflite", "wb").write(tflite_model)