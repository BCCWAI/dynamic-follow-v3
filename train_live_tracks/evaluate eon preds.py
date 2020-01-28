import matplotlib.pyplot as plt
import ast


scales = {'v_ego': [0.0, 30.08179473876953], 'yRel': [-24.200105667114258, 27.49478530883789],
               'steer_rate': [-360.0, 293.0], 'gas': [-0.4711538553237915, 1.0], 'steer_angle': [-506.0, 465.1875],
               'v_lead': [0.0, 30.18574333190918], 'dRel': [0.75, 151.5], 'vRel': [-59.75, 30.0],
               'a_lead': [-5.971687316894531, 9.852273941040039],
               'x_lead': [-16.047168731689453, 169.9757843017578], 'max_tracks': 19}

with open('/Git/dftf_input', 'r') as f:
    d = f.read().split('\n')

data = []
for i in d:
    try:
        data.append(ast.literal_eval(i))
    except:
        pass
inputs = data[::2]
outputs = data[1:][::2]

plt.plot(range(len(outputs)), outputs)
plt.plot(range(len(inputs)), [i[3] for i in inputs])