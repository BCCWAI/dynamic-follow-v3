import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from normalizer import norm
import itertools
os.chdir("C:/Git/dynamic-follow-tf/models/h5_models")
model = tf.keras.models.load_model("3model-epoch-8.h5")
v_scale, a_scale, x_scale = [0.0, 48.288787841797], [-8.39838886261, 9.78254699707], [0.125, 138.625]

data = [norm(0, v_scale), norm(0, v_scale), norm(0, x_scale), norm(0, a_scale)]
prediction=model.predict([[data]])[0][0]
#print(prediction)
print((prediction - 0.5)*2.0)