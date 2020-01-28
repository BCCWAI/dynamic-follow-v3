import json
import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, Activation, LeakyReLU, Flatten
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from keras import backend as K
from sklearn.model_selection import train_test_split
import shutil
import functools
import operator
from keras.models import load_model

def interp_fast(x, xp, fp=[0, 1]):  # extrapolates above range, np.interp does not
    return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]

os.chdir("C:/Git/dynamic-follow-tf-v2")
data_dir = "brake_pred"
model_name = "brake_pred"

def load_scales():
    with open("data/{}/scales".format(data_dir), "rb") as f:
        return pickle.load(f)

def get_scales():
    '''print("Loading data...")
    with open("data/{}/x_train".format(data_dir), "rb") as f:
        x_train = pickle.load(f)'''
    
    #print("Caclulating normalization scales...")
    scales = load_scales()
    return scales

def get_brake_pred_model():
    scales = get_scales()
    return load_model('models/h5_models/{}.h5'.format(model_name)), scales
