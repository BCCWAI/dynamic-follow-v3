import os
import ast
import json
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

os.chdir('D:\Resilio Sync\dfv2\ShaneSmiskol-TOYOTA COROLLA 2017')

def interp_fast(x, xp, fp):  # extrapolates above range, np.interp does not
  return (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]

process_new_data = False
if process_new_data:
    with open('df-data.10', 'r') as f:
        data = [ast.literal_eval(line) for line in f.read().split('\n') if line != '' and line[-1] == '}']
    with open('C:\Git\dynamic-follow-tf-v2\data\\new_data\\new', 'wb') as f:
        pickle.dump(data, f)
else:
    with open('C:\Git\dynamic-follow-tf-v2\data\\new_data\\new', 'rb') as f:
        data = pickle.load(f)

# 0 data: v_lat, status (always True), a_rel (always 0)
# pushed a fix to add live tracks even when no lead ^^^, so status should be fixed with next data
# max brake pressure is 4047
# max gas is 100
#data = [i for i in data if len(i['live_tracks']['tracks']) > 0][10000:25000]
'''x = range(len(data))
y = [i['steer_angle']/50 for i in data]
y2 = [len(i['live_tracks']['tracks']) for i in data]
plt.plot(x, y)
plt.plot(x, y2)
plt.show()'''

#y=[(i['gas'] - 3.8076250553131104) / 262.3935546875 for i in data]
#y = [i if i > 0.005  else 0 for i in y]
'''x = range(len(data))
#pedal_gas = [0 if i['gas'] < 4.88 else i['gas'] for i in data]
pedal_gas = [i['gas'] for i in data]
car_gas = [i['car_gas'] for i in data]

max_car_gas = max(car_gas)  # this is most accurate way to map pedal gas to car gas range (0 to 1)
max_pedal_gas = pedal_gas[car_gas.index(max_car_gas)]

min_car_gas = [i for i in car_gas if i > 0.0][0] #car_gas[len(car_gas)//2]
min_pedal_gas = pedal_gas[car_gas.index(min_car_gas)]

pedal_gas = [interp_fast(i, [min_pedal_gas, max_pedal_gas], [min_car_gas, max_car_gas]) for i in pedal_gas]
pedal_gas = [i if i >= 0 else 0 for i in pedal_gas]

brake = [i['brake'] / 4047 if i['brake'] > 512 and pedal_gas[idx] == 0.0 else 0 for idx, i in enumerate(data)]

#y2 = [1 if i['right_blinker'] else 0 for i in data]
#y2 = [i['steer_angle'] for i in data]
plt.clf()
plt.plot(x, pedal_gas, label='pedal_gas')
plt.plot(x, car_gas, label='car gas')
plt.plot(x, brake, label='brake')
#plt.plot(x, y2, label='steer_angle')
plt.legend()
plt.show()'''

print(len(data))
#track_yRel = [[x['yRel'] for x in i['live_tracks']['tracks']] for i in data]
track_data = [i['live_tracks']['tracks'] for i in data][14500:]
data = data[14500:]

fig = plt.figure(1)
ax = fig.add_subplot(111, projection=None)
xlin = np.linspace(0, 120, 50)

showed = False
for count, i in enumerate(track_data):
    #time.sleep(.00001)
    yRel = []
    vRel = []
    dRel = []
    aRel = []
    vEgo = data[count]['v_ego']
    for x in i:
        yRel.append(x['yRel'])
        vRel.append(data[count]['v_ego'] + x['vRel'])
        dRel.append(x['dRel'])
        aRel.append(x['aRel'])
    
    ax.clear()
    ax.set_autoscale_on(False)
    ax.set_xlim(0, 120)
    ax.set_ylim(-10, 10)
    ax.scatter(dRel, yRel)
    if data[count]['left_blinker']:
        ax.scatter([10], [2.5])
        ax.annotate('left blinker', [10.2, 2.7])
    elif data[count]['left_blinker']:
        ax.scatter([10], [-2.5])
        ax.annotate('left blinker', [10.2, -2.7])
    angle = data[count]['steer_angle']
    y = [(i*angle)**2/(10000*(angle*2)) for i in xlin]
    ax.plot(xlin, y, linestyle='--', label='steer angle: {}'.format(round(angle, 5)))
    #ax.plot([0, 70], [0, 0])
    for idx, coord in enumerate(list(zip(dRel, yRel))):
        ax.annotate('{} m/s, {} a_rel'.format(round(vRel[idx], 3), round(aRel[idx], 3)), [i + .20 for i in coord])
    #ax.annotate('steer angle: {}'.format(angle), [2.0, .2])
    ax.set_xlabel('longitudinal position (m)')
    ax.set_ylabel('lateral position (m)')
    ax.set_title('time step: {}, v_ego: {} ms'.format(count, round(vEgo, 4)))
    ax.legend(bbox_to_anchor=(0, 1.08), loc=2, borderaxespad=0.)
    plt.pause(0.02)
    if not showed:
        showed = True
        plt.show()