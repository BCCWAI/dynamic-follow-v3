'''from numpy.random import seed
seed(255)
from tensorflow import set_random_seed
set_random_seed(255)'''
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
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
import seaborn as sns
from even_out_distribution import even_out_distribution
# from keras.callbacks.tensorboard_v1 import TensorBoard

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1.
# set_session(tf.Session(config=config))
# sns.distplot(data_here)

num_cores = 4
use_gpu = False
if use_gpu:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores,
                        allow_soft_placement=True,
                        device_count={'CPU': num_CPU,
                                      'GPU': num_GPU}
                        )

session = tf.Session(config=config)
K.set_session(session)


def interp_fast(x, xp, fp=[0, 1], ext=False):  # extrapolates above range when ext is True
    interped = (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]
    return interped if ext else min(max(min(fp), interped), max(fp))


# def tanh_estimator(d, scale):
#     d = np.array(d)
#     data = 0.5 * (np.tanh(0.01 * ((d - scale['mean']) / scale['std'])) + 1)
#     # data = np.interp(data, scale['scale'], [-1, 1])
#     return data


os.chdir("C:/Git/dynamic-follow-tf-v2")
data_dir = "simple_model"
norm_dir = "data/{}/normalized"
model_name = "simple_model"

try:
    shutil.rmtree("models/h5_models/{}".format(model_name))
except:
    pass
os.mkdir("models/h5_models/{}".format(model_name))

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


class Visualize(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 1 == 0:
            # accuracy = []
            # for i in range(500):
            #     choice = random.randint(0, len(x_test) - 2)
            #     real = y_test[choice]
            #     to_pred = x_test[choice]
            #     pred = model.predict([[to_pred]])[0][0]
            #     accuracy.append(abs(real - pred))
            #
            # avg = sum(accuracy) / len(accuracy)
            # print("Accuracy: {}".format(abs(avg - 1)))

            scale = [0, 1]
            scale_coast = .5
            dist = 17.8816
            graph_len = 40

            x = [i for i in range(graph_len + 1)]

            # --- DISTANCE ---
            plt.figure(1)
            plt.clf()

            # dist_y = [model.predict(np.asarray([[interp_fast(dist, scales['v_ego']), interp_fast(dist, scales['v_lead']),
            #                                      interp_fast(i, scales['x_lead']), interp_fast(0, scales['a_lead'])]]))[0][0] for i in range(graph_len + 1)]
            dist_y = [
                model.predict(np.asarray([[interp_fast(dist, scales), interp_fast(dist, scales),
                                           interp_fast(i, scales), interp_fast(0, scales)]]))[0][0]
                for i in range(graph_len + 1)]


            plt.plot([0, graph_len], [scale_coast, scale_coast], '--', linewidth=1)
            plt.plot([dist, dist], scale, '--', linewidth=1)

            plt.plot(x, dist_y, label='epoch-{}'.format(epoch))
            plt.title('distance')
            plt.legend()
            plt.savefig("models/h5_models/{}/1-{}-epoch-{}.png".format(model_name, model_name, epoch))
            plt.pause(.01)

            # --- VELOCITY ---
            plt.figure(2)
            plt.clf()

            # vel_y = [model.predict(np.asarray([[interp_fast(dist, scales['v_ego']), interp_fast(i, scales['v_lead']),
            #                                     interp_fast(dist, scales['x_lead']), interp_fast(0, scales['a_lead'])]]))[0][0] for i in range(graph_len + 1)]
            vel_y = [model.predict(np.asarray([[interp_fast(dist, scales), interp_fast(i, scales),
                                                interp_fast(dist, scales),
                                                interp_fast(0, scales)]]))[0][0] for i in
                     range(graph_len + 1)]


            plt.plot([0, graph_len], [scale_coast, scale_coast], '--', linewidth=1)
            plt.plot([dist, dist], scale, '--', linewidth=1)

            plt.plot(x, vel_y, label='epoch-{}'.format(epoch))
            plt.title('velocity')
            plt.legend()
            plt.savefig("models/h5_models/{}/2-{}-epoch-{}.png".format(model_name, model_name, epoch))
            plt.pause(.01)

            # # --- GROUND TRUTH ---
            # plt.figure(3)
            # plt.clf()
            #
            # pred_num = 100
            #
            # x = range(pred_num)
            # ground_y = [i for i in y_test[:pred_num]]
            #
            # pred_y = [model.predict(np.asarray([i]))[0][0] for i in x_test[:pred_num]]
            #
            # plt.plot(x, ground_y, label='ground-truth')
            # plt.plot(x, pred_y, label='epoch-{}'.format(epoch))
            # plt.title("ground truths")
            # plt.legend()
            # plt.savefig("models/h5_models/{}/3-{}-epoch-{}.png".format(model_name, model_name, epoch))
            #
            # plt.pause(.01)
            model.save("models/h5_models/{}/{}-epoch-{}.h5".format(model_name, model_name, epoch))
            '''if epoch - 1 % 2 == 0 and epoch!=0:
                print("Stop training?")
                stop = input("[Y/n]: ")
                if stop.lower()=="y":
                    model.stop_training = True'''


force_normalize = False
if os.path.exists("data/{}/x_train_normalized".format(data_dir)) and not force_normalize:
    print('Loading normalized data...', flush=True)
    with open("data/{}/x_train_normalized".format(data_dir), "rb") as f:
        x_train, scales = pickle.load(f)
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

    remove_brake = False
    if remove_brake:
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if y >= 0.0])
        x_train, y_train = map(list, [x_train, y_train])

    replace_brake_with_coast = False
    if replace_brake_with_coast:
        y_train = [sample if sample >= 0.0 else 0.0 for sample in y_train]

    only_leads = False
    if only_leads:
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if x['lead_status']])
        x_train, y_train = map(list, [x_train, y_train])

    filter_low_speeds = True
    if filter_low_speeds:
        print(len(x_train))
        mph_to_ms = 0.44704
        min_speed = 1 * mph_to_ms
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if x['v_ego'] >= min_speed])
        x_train, y_train = map(list, [x_train, y_train])
        print(len(x_train))

    distance_filter = False
    if distance_filter:
        print(len(x_train))
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if x['x_lead'] < 60])
        x_train, y_train = map(list, [x_train, y_train])
        print(len(x_train))

    filter_high_speeds = False
    if filter_high_speeds:
        mph_to_ms = 0.44704
        min_speed = 40 * mph_to_ms
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if x['v_ego'] <= min_speed])
        x_train, y_train = map(list, [x_train, y_train])

    bad_data_filter = False
    if bad_data_filter:
        print(len(x_train))
        x_train, y_train = zip(*[[x, y] for x, y in zip(x_train, y_train) if not (x['v_lead'] - x['v_ego'] < -0.44704 and y > .0)])
        x_train, y_train = map(list, [x_train, y_train])
        print(len(x_train))

    model_inputs = ['v_ego', 'v_lead', 'x_lead', 'a_lead']

    x_train, y_train = even_out_distribution(x_train, y_train, n_sections=15, m=2, reduction=.4, reduce_min=.5)
    print("Normalizing data...", flush=True)
    x_train, scales = normX(x_train, model_inputs)

    y_train = np.array(y_train)
    print('Dumping normalized data...', flush=True)
    with open("data/{}/x_train_normalized".format(data_dir), "wb") as f:
        pickle.dump([x_train, scales], f)
    with open("data/{}/y_train_normalized".format(data_dir), "wb") as f:
        pickle.dump(y_train, f)

# scales['gas'] = [min(y_train), max(y_train)]
# y_train = np.interp(y_train, [0, 1], [-1, 1])
# y_train = np.interp(y_train, [-1, 1], [0, 1])  # this is the best performing model architecture

y_train = np.clip((y_train - 0.12) * 1.4, -1, 1)  # subtract offset caused by regen from chevy
y_train = np.interp(y_train, [-1, 1], [0, 1])


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.07)
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

# opt = keras.optimizers.Adam(lr=0.0001)
# opt = keras.optimizers.Adadelta(lr=2) #lr=.000375)
# opt = keras.optimizers.SGD(lr=0.008, momentum=0.9)
# opt = keras.optimizers.RMSprop(lr=0.01)#, decay=1e-5)
# opt = keras.optimizers.Adagrad(lr=0.00025)
opt = keras.optimizers.Adagrad()
# opt = 'adam'

# opt = 'rmsprop'
# opt = keras.optimizers.Adadelta()

layer_num = 6
nodes = 346
a_function = "relu"

model = Sequential()
model.add(Dense(64, activation=a_function, input_shape=(x_train.shape[1:])))
model.add(Dense(32, activation=a_function))
model.add(Dense(1))

model.compile(loss='mse', optimizer=opt, metrics=['mae'])

# tensorboard = TensorBoard(log_dir="C:/Git/dynamic-follow-tf-v2/train_model/logs/{}".format("final model"))
callbacks = [Visualize()]
model.fit(x_train, y_train,
          shuffle=True,
          batch_size=64,
          epochs=10000,
          validation_data=(x_test, y_test),
          callbacks=callbacks)

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


def visualize(to_display=500):
    try:
        plt.figure(0)
        plt.clf()
        plt.title('coast samples: predicted vs ground')
        find = .0
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
        find = .2
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
        find = .4
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
        find = -.1
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
        find = -0.25
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


preds = model.predict([[x_test]]).reshape(1, -1)
diffs = [abs(pred - ground) for pred, ground in zip(preds[0], y_test)]

print("Test accuracy: {}".format(interp_fast(sum(diffs) / len(diffs), [0, 1], [1, 0], ext=True)))

for i in range(20):
    c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict([x_test[c]])[0][0]))
    print()

for c in np.where(y_test==0.0)[0][:20]:
    #c = random.randint(0, len(x_test))
    print('Ground truth: {}'.format(y_test[c]))
    print('Prediction: {}'.format(model.predict([[x_test[c]]])[0][0]))
    print()


def coast_test():
    coast_samples = np.where(y_test == 0.0)[0]
    coast_predictions = model.predict(x_test[(coast_samples)])
    num_invalid = 0
    for i in coast_predictions:
        if i[0] >= .05:
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