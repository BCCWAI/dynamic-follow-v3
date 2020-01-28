import matplotlib.pyplot as plt
import json
import ast
import numpy as np
from scipy import interpolate

with open('C:\Git\dynamic-follow-tf\df_output', 'r') as f:
    data = f.read()
scales = {'v_ego_scale': [0.0, 40.755523681641],
              'v_lead_scale': [0.0, 44.508262634277],
              'x_lead_scale': [0.125, 146.375]}
              
data = data.split('\n')[1:]

preds = [float(i[:i.index(' ')]) for i in data]
data = [json.loads(i[i.index(' ') + 1:]) for i in data]
flattened = [val for sublist in data for val in sublist]
v_ego = [i[-1][0] for i in data]
v_lead = [i[-1][1] for i in data]
x_lead = [i[-1][2] for i in data]

v_ego_i = interpolate.interp1d([0,1], scales['v_ego_scale'], fill_value='extrapolate')
v_lead_i = interpolate.interp1d([0,1], scales['v_lead_scale'], fill_value='extrapolate')
x_lead_i= interpolate.interp1d([0,1], scales['x_lead_scale'], fill_value='extrapolate')

#v_ego_o = interpolate.interp1d([0.0, 39.129379272461], [0,1], fill_value='extrapolate')
#v_lead_o = interpolate.interp1d([0,1], [0.0, 44.459167480469], fill_value='extrapolate')
#x_lead_o = interpolate.interp1d([0.375, 146.375], [0,1], fill_value='extrapolate')

plt.clf()
plt.plot(range(len(data))[20000:30000], v_ego[20000:30000])
plt.plot(range(len(data))[20000:30000], v_lead[20000:30000])
#plt.plot(range(len(data)), [x_lead_i(i) for i in x_lead])

#plt.plot(range(len(data))[20000:30000], [v_ego_i(i) for i in v_ego][20000:30000])
#plt.plot(range(len(data))[20000:30000], [v_lead_i(i) for i in v_lead][20000:30000])
plt.plot(range(len(data))[20000:30000], preds[20000:30000])

#plt.plot(range(len(data)), [v_ego_o(0) for i in range(len(data))])
plt.show()